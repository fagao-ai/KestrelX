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
export type StreamingModelHandler = (
  request: ModelRequest,
  emit: (delta: ModelDelta) => void
) => Promise<ModelResponse>;

export interface SessionStore {
  load(sessionId: string): Promise<Record<string, unknown> | null> | Record<string, unknown> | null;
  save(session: Record<string, unknown>): Promise<void> | void;
  delete?(sessionId: string): Promise<void> | void;
}

export interface Memory {
  recall(query: Record<string, unknown>): Promise<unknown[]> | unknown[];
  remember(entry: Record<string, unknown>): Promise<void> | void;
}

export interface AgentOptions {
  model?: ModelHandler;
  streamModel?: StreamingModelHandler;
  onEvent?: EventHandler;
  eventHandler?: EventHandler;
  sessionStore?: SessionStore;
  memory?: Memory;
}

export interface McpServerConfig {
  url?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  headers?: Record<string, string>;
  cwd?: string;
  request_timeout_seconds?: number;
  protocol_version?: string;
  client_name?: string;
  client_version?: string;
}

export declare function contractVersion(): number;
export declare function capabilities(): string[];
export declare function validateConfig(
  config: Record<string, unknown>,
  hasCustomModel?: boolean
): void;

export declare function createAgent(
  config: Record<string, unknown>,
  model?: ModelHandler,
  eventHandler?: EventHandler
): Agent;
export declare function createAgent(
  config: Record<string, unknown>,
  options?: AgentOptions
): Agent;
export declare function createAgentAsync(
  config: Record<string, unknown>,
  model?: ModelHandler,
  eventHandler?: EventHandler
): Promise<Agent>;
export declare function createAgentAsync(
  config: Record<string, unknown>,
  options?: AgentOptions
): Promise<Agent>;

export declare class Agent {
  run(prompt: string): Promise<AgentOutput>;
  startRun(prompt: string): Run;
  session(): Promise<Record<string, unknown>>;
  todos(): unknown[] | null;
  reset(): Promise<void>;
  deleteSession(): Promise<void>;
  setEventHandler(handler?: EventHandler): void;
  registerTool(spec: ToolSpec, handler: ToolHandler): void;
  connectMcp(servers: McpServerConfig[]): Promise<void>;
  closeMcp(): Promise<void>;
  close(): void;
}

export declare class Run {
  cancel(): void;
  isDone(): boolean;
  result(): Promise<AgentOutput>;
}

export declare function kestrelVersion(): string;
