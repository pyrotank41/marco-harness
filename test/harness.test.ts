import { describe, it, expect } from 'vitest'
import { z } from 'zod'
import { Harness } from '../src/harness.js'
import { MockProvider } from '../src/providers/mock.js'
import type { Tool } from '../src/tools.js'
import type { AssistantMessage } from '../src/messages.js'
import type { Hooks } from '../src/hooks.js'

const echoTool: Tool = {
  name: 'echo',
  description: 'Echo input',
  inputJsonSchema: {
    type: 'object', properties: { text: { type: 'string' } }, required: ['text'],
  },
  validate: (i) => z.object({ text: z.string() }).parse(i),
  handler: async (input) => `got ${(input as { text: string }).text}`,
}

describe('Harness', () => {
  it('runs a single-turn conversation and returns the final message', async () => {
    const endMsg: AssistantMessage = {
      role: 'assistant', text: 'hi!', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const harness = new Harness({
      provider: new MockProvider([[{ type: 'message_end', message: endMsg }]]),
      modelConfig: { model: 'mock' },
    })
    const result = await harness.run({ kind: 'user_message', text: 'hello' })
    expect(result.status).toBe('completed')
    expect(result.finalMessage?.text).toBe('hi!')
  })

  it('rejects a run when onRunStart returns allowed=false', async () => {
    const endMsg: AssistantMessage = {
      role: 'assistant', text: 'never called', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const hooks: Hooks = {
      onRunStart: async ({ messages }) => ({ allowed: false, rejectReason: 'blocked', messages }),
    }
    const harness = new Harness({
      provider: new MockProvider([[{ type: 'message_end', message: endMsg }]]),
      modelConfig: { model: 'mock' },
      hooks,
    })
    const result = await harness.run({ kind: 'user_message', text: 'x' })
    expect(result.status).toBe('aborted')
    expect(result.abortReason).toBe('blocked')
  })

  it('allows onRunStart to add a system prompt and hydrate messages', async () => {
    const endMsg: AssistantMessage = {
      role: 'assistant', text: 'ok', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const hooks: Hooks = {
      onRunStart: async ({ messages }) => ({
        allowed: true,
        messages: [
          { role: 'system', text: 'you are echo bot' },
          ...messages,
        ],
      }),
    }
    const harness = new Harness({
      provider: new MockProvider([[{ type: 'message_end', message: endMsg }]]),
      modelConfig: { model: 'mock' },
      hooks,
    })
    const result = await harness.run({ kind: 'user_message', text: 'hi' })
    expect(result.messages[0].role).toBe('system')
    expect(result.messages[1].role).toBe('user')
  })

  it('invokes onRunEnd with the final status', async () => {
    const endMsg: AssistantMessage = {
      role: 'assistant', text: 'done', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 1, outputTokens: 1 },
    }
    let captured: { status: string; iterations: number } | undefined
    const hooks: Hooks = {
      onRunEnd: async ({ status, iterations }) => {
        captured = { status, iterations }
      },
    }
    const harness = new Harness({
      provider: new MockProvider([[{ type: 'message_end', message: endMsg }]]),
      modelConfig: { model: 'mock' },
      hooks,
    })
    await harness.run({ kind: 'user_message', text: 'x' })
    expect(captured?.status).toBe('completed')
    expect(captured?.iterations).toBe(1)
  })

  it('invokes onRunEnd with truncated when the model stops on max_tokens', async () => {
    const clipped: AssistantMessage = {
      role: 'assistant', text: 'partial', toolCalls: [],
      stopReason: 'max_tokens', usage: { inputTokens: 1, outputTokens: 1 },
    }
    let captured: { status: string } | undefined
    const hooks: Hooks = {
      onRunEnd: async ({ status }) => {
        captured = { status }
      },
    }
    const harness = new Harness({
      provider: new MockProvider([[{ type: 'message_end', message: clipped }]]),
      modelConfig: { model: 'mock' },
      hooks,
    })
    const result = await harness.run({ kind: 'user_message', text: 'x' })
    expect(result.status).toBe('truncated')
    expect(captured?.status).toBe('truncated')
  })

  it('supports registering tools', async () => {
    const endMsg: AssistantMessage = {
      role: 'assistant', text: 'done', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const harness = new Harness({
      provider: new MockProvider([[{ type: 'message_end', message: endMsg }]]),
      modelConfig: { model: 'mock' },
    })
    harness.registerTool(echoTool)
    expect(harness.getTool('echo')).toBe(echoTool)
  })

  // Tool-execution tests — these exercise the Harness's tool execution
  // logic (beforeToolCall → validate → handler → afterToolResult), which
  // used to live in the inner loop but is now correctly a harness concern.

  it('denies a tool call via beforeToolCall and returns the reason as an error tool result', async () => {
    const turn1: AssistantMessage = {
      role: 'assistant', text: '', toolCalls: [
        { id: 'c1', name: 'echo', input: { text: 'nope' } },
      ],
      stopReason: 'tool_use', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const turn2: AssistantMessage = {
      role: 'assistant', text: 'ok sorry', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 2, outputTokens: 1 },
    }
    const provider = new MockProvider([
      [{ type: 'message_end', message: turn1 }],
      [{ type: 'message_end', message: turn2 }],
    ])
    const hooks: Hooks = {
      beforeToolCall: async ({ toolCall }) =>
        toolCall.name === 'echo'
          ? { decision: 'deny', reason: 'not allowed' }
          : { decision: 'execute' },
    }
    const harness = new Harness({
      provider,
      modelConfig: { model: 'mock' },
      hooks,
      tools: [echoTool],
    })
    const result = await harness.run({ kind: 'user_message', text: 'nope' })

    const toolMsg = result.messages[2]
    expect(toolMsg.role).toBe('tool')
    if (toolMsg.role === 'tool') {
      expect(toolMsg.isError).toBe(true)
      expect(toolMsg.content).toContain('not allowed')
    }
  })

  it('short-circuits a tool call and returns the provided result', async () => {
    const turn1: AssistantMessage = {
      role: 'assistant', text: '', toolCalls: [
        { id: 'c1', name: 'echo', input: { text: 'x' } },
      ],
      stopReason: 'tool_use', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const turn2: AssistantMessage = {
      role: 'assistant', text: 'done', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 2, outputTokens: 1 },
    }
    const provider = new MockProvider([
      [{ type: 'message_end', message: turn1 }],
      [{ type: 'message_end', message: turn2 }],
    ])
    const hooks: Hooks = {
      beforeToolCall: async () => ({ decision: 'short-circuit', result: 'CACHED' }),
    }
    const harness = new Harness({
      provider,
      modelConfig: { model: 'mock' },
      hooks,
      tools: [echoTool],
    })
    const result = await harness.run({ kind: 'user_message', text: 'x' })

    const toolMsg = result.messages[2]
    expect(toolMsg.role).toBe('tool')
    if (toolMsg.role === 'tool') {
      expect(toolMsg.isError).toBe(false)
      expect(toolMsg.content).toBe('CACHED')
    }
  })

  it('returns an error tool result when the model calls an unknown tool', async () => {
    const turn1: AssistantMessage = {
      role: 'assistant', text: '', toolCalls: [
        { id: 'c1', name: 'unknown_tool', input: {} },
      ],
      stopReason: 'tool_use', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const turn2: AssistantMessage = {
      role: 'assistant', text: 'oh well', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 2, outputTokens: 1 },
    }
    const provider = new MockProvider([
      [{ type: 'message_end', message: turn1 }],
      [{ type: 'message_end', message: turn2 }],
    ])
    const harness = new Harness({
      provider,
      modelConfig: { model: 'mock' },
    })
    const result = await harness.run({ kind: 'user_message', text: '' })

    const toolMsg = result.messages[2]
    expect(toolMsg.role).toBe('tool')
    if (toolMsg.role === 'tool') {
      expect(toolMsg.isError).toBe(true)
      expect(toolMsg.content).toContain('not registered')
    }
  })

  it('runs afterToolResult after a successful tool execution and uses its transformed result', async () => {
    const turn1: AssistantMessage = {
      role: 'assistant', text: '', toolCalls: [
        { id: 'c1', name: 'echo', input: { text: 'secret' } },
      ],
      stopReason: 'tool_use', usage: { inputTokens: 1, outputTokens: 1 },
    }
    const turn2: AssistantMessage = {
      role: 'assistant', text: 'done', toolCalls: [],
      stopReason: 'end_turn', usage: { inputTokens: 2, outputTokens: 1 },
    }
    const provider = new MockProvider([
      [{ type: 'message_end', message: turn1 }],
      [{ type: 'message_end', message: turn2 }],
    ])
    const hooks: Hooks = {
      afterToolResult: async ({ result }) => ({ result: result.replace(/secret/g, '<redacted>') }),
    }
    const harness = new Harness({
      provider,
      modelConfig: { model: 'mock' },
      hooks,
      tools: [echoTool],
    })
    const result = await harness.run({ kind: 'user_message', text: 'x' })

    const toolMsg = result.messages[2]
    expect(toolMsg.role).toBe('tool')
    if (toolMsg.role === 'tool') {
      expect(toolMsg.content).toBe('got <redacted>')
      expect(toolMsg.isError).toBe(false)
    }
  })
})
