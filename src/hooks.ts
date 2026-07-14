// MARCO — lifecycle hook types and runner. Five hooks map 1:1 to nodes in the outer-loop diagram.

import type { Message, AssistantMessage, ToolCall, ToolResultMessage, UserMessageContentPart } from './messages.js'
import type { ModelConfig } from './provider.js'

export type Trigger =
  | {
      kind: 'user_message'
      text: string
      // Optional multimodal payload. Forwarded onto the synthesized
      // UserMessage.content. When set, providers render multipart
      // (images, documents) natively for capable models.
      content?: UserMessageContentPart[]
      metadata?: Record<string, unknown>
    }
  | { kind: 'schedule'; cron: string; metadata?: Record<string, unknown> }
  | { kind: 'event'; eventType: string; payload: unknown }
  | { kind: 'webhook'; path: string; payload: unknown }

export type OnRunStartInput = {
  trigger: Trigger
  runId: string
  messages: Message[]
  modelConfig: ModelConfig
}

export type OnRunStartOutput = {
  allowed: boolean
  rejectReason?: string
  messages: Message[]
  modelConfig?: ModelConfig
}

export type BeforeModelCallInput = {
  messages: Message[]
  iteration: number
  runId: string
}

export type BeforeModelCallOutput = {
  messages: Message[]
  modelConfig?: ModelConfig
  abort?: boolean
  abortReason?: string
}

export type BeforeToolCallInput = {
  toolCall: ToolCall
  runId: string
}

export type BeforeToolCallOutput =
  | { decision: 'execute'; input?: unknown }
  | { decision: 'deny'; reason: string }
  | { decision: 'short-circuit'; result: string }

export type AfterToolResultInput = {
  toolCall: ToolCall
  result: string
  isError: boolean
  durationMs: number
  runId: string
}

export type AfterToolResultOutput = {
  result: string
  isError?: boolean
}

export type OnRunEndInput = {
  runId: string
  status: 'completed' | 'truncated' | 'aborted' | 'errored'
  finalMessage?: AssistantMessage
  messages: Message[]
  error?: Error
  iterations: number
}

export type Hooks = {
  onRunStart?: (input: OnRunStartInput) => Promise<OnRunStartOutput>
  beforeModelCall?: (input: BeforeModelCallInput) => Promise<BeforeModelCallOutput>
  beforeToolCall?: (input: BeforeToolCallInput) => Promise<BeforeToolCallOutput>
  afterToolResult?: (input: AfterToolResultInput) => Promise<AfterToolResultOutput>
  onRunEnd?: (input: OnRunEndInput) => Promise<void>
}

export async function runHook<I, O>(
  hook: ((input: I) => Promise<O>) | undefined,
  input: I,
): Promise<O | undefined> {
  if (!hook) return undefined
  return hook(input)
}
