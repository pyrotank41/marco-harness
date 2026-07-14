import { describe, it, expect } from 'vitest'
import { z } from 'zod'
import { runInnerLoop, type RunInnerLoopInput } from '../src/innerLoop.js'
import { MockProvider } from '../src/providers/mock.js'
import { ToolRegistry, type Tool } from '../src/tools.js'
import type { AssistantMessage, ToolCall, ToolResultMessage } from '../src/messages.js'

const echoTool: Tool = {
  name: 'echo',
  description: 'Echo input back',
  inputJsonSchema: {
    type: 'object',
    properties: { text: { type: 'string' } },
    required: ['text'],
  },
  validate: (i) => z.object({ text: z.string() }).parse(i),
  handler: async (input) => `echoed: ${(input as { text: string }).text}`,
}

/**
 * Minimal stand-in for the harness's requestToolCall function. The inner
 * loop tests don't exercise hook-driven execution paths (deny,
 * short-circuit, unknown tool) — those are the harness's responsibility
 * and live in test/harness.test.ts. Here we just need to actually run
 * the tool when the loop asks for it.
 */
function makeDirectExecutor(registry: ToolRegistry) {
  return async (call: ToolCall): Promise<ToolResultMessage> => {
    const tool = registry.get(call.name)
    if (!tool) {
      return { role: 'tool', toolCallId: call.id, content: 'not registered', isError: true }
    }
    try {
      const validated = tool.validate(call.input)
      const result = await tool.handler(validated, { runId: 'test' })
      return { role: 'tool', toolCallId: call.id, content: result, isError: false }
    } catch (err) {
      return { role: 'tool', toolCallId: call.id, content: String(err), isError: true }
    }
  }
}

const noExecutor = async (): Promise<ToolResultMessage> => {
  throw new Error('requestToolCall should not be called in this test')
}

describe('runInnerLoop', () => {
  it('returns on end_turn without calling any tool', async () => {
    const endMsg: AssistantMessage = {
      role: 'assistant', text: 'done', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const provider = new MockProvider([[{ type: 'message_end', message: endMsg }]])
    const input: RunInnerLoopInput = {
      runId: 'r1',
      messages: [{ role: 'user', text: 'hi' }],
      provider,
      toolSpecs: [],
      requestToolCall: noExecutor,
      hooks: {},
      modelConfig: { model: 'mock' },
    }
    const result = await runInnerLoop(input)
    expect(result.status).toBe('completed')
    expect(result.finalMessage).toEqual(endMsg)
    expect(result.iterations).toBe(1)
  })

  it('returns truncated when the model stops on max_tokens', async () => {
    const clipped: AssistantMessage = {
      role: 'assistant', text: 'partial answ', toolCalls: [],
      stopReason: 'max_tokens', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const provider = new MockProvider([[{ type: 'message_end', message: clipped }]])
    const result = await runInnerLoop({
      runId: 'r-trunc',
      messages: [{ role: 'user', text: 'hi' }],
      provider,
      toolSpecs: [],
      requestToolCall: noExecutor,
      hooks: {},
      modelConfig: { model: 'mock' },
    })
    expect(result.status).toBe('truncated')
    expect(result.finalMessage).toEqual(clipped)
    expect(result.iterations).toBe(1)
  })

  it('delegates tool calls to requestToolCall and feeds results back until end_turn', async () => {
    const turn1: AssistantMessage = {
      role: 'assistant', text: '', toolCalls: [
        { id: 'c1', name: 'echo', input: { text: 'hi' } },
      ],
      stopReason: 'tool_use', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const turn2: AssistantMessage = {
      role: 'assistant', text: 'all done', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 2, outputTokens: 1 },
    }
    const provider = new MockProvider([
      [{ type: 'message_end', message: turn1 }],
      [{ type: 'message_end', message: turn2 }],
    ])
    const registry = new ToolRegistry()
    registry.register(echoTool)

    const result = await runInnerLoop({
      runId: 'r2',
      messages: [{ role: 'user', text: 'say hi' }],
      provider,
      toolSpecs: registry.toSpecs(),
      requestToolCall: makeDirectExecutor(registry),
      hooks: {},
      modelConfig: { model: 'mock' },
    })

    expect(result.status).toBe('completed')
    expect(result.iterations).toBe(2)

    const roles = result.messages.map((m) => m.role)
    expect(roles).toEqual(['user', 'assistant', 'tool', 'assistant'])
    const toolMsg = result.messages[2]
    expect(toolMsg.role).toBe('tool')
    if (toolMsg.role === 'tool') {
      expect(toolMsg.content).toContain('echoed: hi')
    }
  })

  it('aborts when beforeModelCall returns abort=true', async () => {
    const endMsg: AssistantMessage = {
      role: 'assistant', text: 'never reached', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const provider = new MockProvider([[{ type: 'message_end', message: endMsg }]])

    const result = await runInnerLoop({
      runId: 'r5',
      messages: [{ role: 'user', text: 'hi' }],
      provider,
      toolSpecs: [],
      requestToolCall: noExecutor,
      hooks: {
        beforeModelCall: async ({ messages }) => ({ messages, abort: true, abortReason: 'budget' }),
      },
      modelConfig: { model: 'mock' },
    })

    expect(result.status).toBe('aborted')
    expect(result.iterations).toBe(0)
  })
})
