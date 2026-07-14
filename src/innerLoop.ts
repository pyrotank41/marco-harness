// MARCO — inner loop. The engine: build context, call model, request tool
// execution, accumulate results, decide stop.
//
// Deliberately small. The loop does NOT execute tools itself — it calls
// the `requestToolCall` function supplied by the harness. The loop's
// world is narrow: provider, tool specs (as data), beforeModelCall hook,
// and requestToolCall. Everything else about tools — registry, handlers,
// permission gates, redaction — lives in the harness layer.
//
// Cancellation: the loop accepts an optional AbortSignal. When the signal
// fires, every active `await` in the loop becomes a cancellation point —
// `provider.stream` receives the signal and its underlying fetch aborts;
// hook + tool calls are wrapped in `raceWithAbort`. There are NO explicit
// `signal.aborted` checks in the loop body. The signal CAUSES unwinds via
// AbortError, which the outer try/catch translates into a clean
// `{ status: 'aborted' }` return.

import type { AssistantMessage, Message, ToolCall, ToolResultMessage } from './messages.js'
import type { ModelConfig, ModelProvider, ToolSpec } from './provider.js'
import { runHook, type Hooks } from './hooks.js'

export type RequestToolCallOptions = {
  signal?: AbortSignal
}

export type RunInnerLoopInput = {
  runId: string
  messages: Message[]
  provider: ModelProvider
  toolSpecs: ToolSpec[]
  requestToolCall: (call: ToolCall, options?: RequestToolCallOptions) => Promise<ToolResultMessage>
  hooks: Pick<Hooks, 'beforeModelCall'>
  modelConfig: ModelConfig
  maxIterations?: number
  /** Cancel the run. When fired, the in-flight fetch (model call) and
   *  the in-flight tool/hook awaits unwind via AbortError; the loop
   *  returns { status: 'aborted', messages, abortReason }. */
  signal?: AbortSignal
}

export type RunInnerLoopResult = {
  status: 'completed' | 'truncated' | 'aborted' | 'errored'
  finalMessage?: AssistantMessage
  messages: Message[]
  iterations: number
  error?: Error
  abortReason?: string
}

const DEFAULT_MAX_ITERATIONS = 25

export async function runInnerLoop(input: RunInnerLoopInput): Promise<RunInnerLoopResult> {
  const { runId, provider, toolSpecs, requestToolCall, hooks, modelConfig, signal } = input
  const maxIterations = input.maxIterations ?? DEFAULT_MAX_ITERATIONS

  let messages: Message[] = [...input.messages]
  let iteration = 0
  let config = modelConfig

  try {
    while (iteration < maxIterations) {
      // Phase 1 — apply harness overrides (beforeModelCall hook).
      // Wrapped in raceWithAbort so a slow hook can be cancelled.
      const harnessOverrides = await raceWithAbort(
        runHook(hooks.beforeModelCall, { messages, iteration, runId }),
        signal,
      )
      if (harnessOverrides?.abort) {
        return {
          status: 'aborted',
          messages: harnessOverrides.messages,
          iterations: iteration,
          abortReason: harnessOverrides.abortReason,
        }
      }
      messages = harnessOverrides?.messages ?? messages
      config = harnessOverrides?.modelConfig ?? config

      // Phase 2 — call the model. Signal threaded into provider.stream;
      // a signal-aware provider's underlying fetch will throw AbortError
      // mid-stream when the signal fires, which propagates up to our
      // outer catch and returns `aborted`.
      let assistantMessage: AssistantMessage | undefined
      for await (const event of provider.stream(messages, toolSpecs, config, { signal })) {
        if (event.type === 'message_end') assistantMessage = event.message
      }
      if (!assistantMessage) {
        return {
          status: 'errored',
          messages,
          iterations: iteration,
          error: new Error('Provider stream ended without message_end event'),
        }
      }

      messages = [...messages, assistantMessage]
      iteration += 1

      // Phase 3 — route on stop reason
      switch (assistantMessage.stopReason) {
        case 'end_turn':
          return { status: 'completed', finalMessage: assistantMessage, messages, iterations: iteration }

        // The model was cut off by the output-token limit, not finished.
        // A distinct status lets the caller tell a clipped answer from a
        // complete one instead of silently reporting success.
        case 'max_tokens':
          return { status: 'truncated', finalMessage: assistantMessage, messages, iterations: iteration }

        case 'error':
        case 'safety':
          return {
            status: 'errored',
            finalMessage: assistantMessage,
            messages,
            iterations: iteration,
            error: new Error(`stopReason=${assistantMessage.stopReason}`),
          }

        case 'tool_use': {
          // The loop requests; the harness handles. Tool calls are raced
          // against the signal — when aborted, raceWithAbort rejects
          // immediately even though the tool's promise keeps running
          // (orphaned). Tools that need cleanup can listen to ctx.abortSignal
          // (threaded through requestToolCall → handleToolCall → handler).
          const toolResults: ToolResultMessage[] = []
          for (const call of assistantMessage.toolCalls) {
            toolResults.push(await raceWithAbort(requestToolCall(call, { signal }), signal))
          }
          messages = [...messages, ...toolResults]
          break
        }
      }
    }

    return {
      status: 'aborted',
      messages,
      iterations: iteration,
      abortReason: `exceeded maxIterations (${maxIterations})`,
    }
  } catch (err) {
    if (isAbortError(err) || signal?.aborted) {
      return {
        status: 'aborted',
        messages,
        iterations: iteration,
        abortReason: signalReason(signal) ?? 'aborted',
      }
    }
    return {
      status: 'errored',
      messages,
      iterations: iteration,
      error: err instanceof Error ? err : new Error(String(err)),
    }
  }
}

/**
 * Race a promise against an abort signal. If the signal fires, the
 * returned promise rejects with AbortError on the next microtask. The
 * underlying promise keeps running but its result is ignored.
 *
 * Listener cleanup: registers `addEventListener('abort', ...)` once per
 * call and removes it on settle so we don't leak listeners across loop
 * iterations.
 */
export function raceWithAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise
  if (signal.aborted) {
    return Promise.reject(makeAbortError(signal))
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(makeAbortError(signal))
    signal.addEventListener('abort', onAbort, { once: true })
    promise
      .then(
        (v) => {
          signal.removeEventListener('abort', onAbort)
          resolve(v)
        },
        (e) => {
          signal.removeEventListener('abort', onAbort)
          reject(e)
        },
      )
  })
}

function makeAbortError(signal: AbortSignal): Error {
  const reason = signalReason(signal) ?? 'aborted'
  // DOMException is the Web standard; falls back to a plain Error in
  // environments where DOMException isn't available.
  if (typeof DOMException !== 'undefined') {
    return new DOMException(reason, 'AbortError')
  }
  const err = new Error(reason)
  err.name = 'AbortError'
  return err
}

function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === 'AbortError'
}

function signalReason(signal: AbortSignal | undefined): string | undefined {
  if (!signal) return undefined
  const r = signal.reason
  if (r === undefined) return undefined
  if (r instanceof Error) return r.message
  if (typeof r === 'string') return r
  try { return String(r) } catch { return undefined }
}
