export interface Usage {
  input_tokens: number;
  output_tokens: number;
}

export interface ContentPart {
  type: 'text' | 'image';
  text?: string;
  url?: string;
  mime_type?: string;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: ContentPart[];
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
}

export interface ToolSpec {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export interface ModelRequest {
  messages: Message[];
  tools?: ToolSpec[];
  tool_choice?: unknown;
  temperature?: number;
  max_tokens?: number;
}

export interface ModelResponse {
  message: Message;
  finish_reason: string;
  usage?: Usage;
  model?: string;
  raw?: unknown;
}

export interface ToolOutput {
  content: string;
  is_error?: boolean;
}

export interface AgentOutput {
  run_id: string;
  session_id: string;
  message: Message;
  text: string;
  turns: number;
  usage: Usage;
  finish_reason: unknown;
}

export interface ModelDelta {
  type: 'text' | 'tool_call';
  text?: string;
  index?: number;
  id?: string;
  name?: string;
  arguments?: string;
}

export interface AgentEvent {
  type: string;
  run_id: string;
  session_id: string;
  sequence: number;
  turn?: number;
  delta?: ModelDelta;
  [key: string]: unknown;
}

export type ModelHandler = (request: ModelRequest) => Promise<ModelResponse>;
export type ToolHandler = (
  callId: string,
  arguments_: Record<string, unknown>
) => Promise<ToolOutput | string>;
export type EventHandler = (event: AgentEvent) => void;

export declare function createAgent(
  config: Record<string, unknown>,
  model?: ModelHandler,
  eventHandler?: EventHandler
): Agent;

export declare class Agent {
  run(prompt: string): Promise<AgentOutput>;
  session(): Promise<Record<string, unknown>>;
  todos(): unknown[] | null;
  reset(): Promise<void>;
  setEventHandler(handler?: EventHandler): void;
  registerTool(spec: ToolSpec, handler: ToolHandler): void;
  close(): void;
}

export declare function kestrelVersion(): string;
