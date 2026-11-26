package kestrel

import "encoding/json"

// Role is a conversation message role.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentPart is one text or image message part.
type ContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	URL      string `json:"url,omitempty"`
	MIMEType string `json:"mime_type,omitempty"`
}

// TextPart creates a text content part.
func TextPart(text string) ContentPart {
	return ContentPart{Type: "text", Text: text}
}

// ImagePart creates an image URL content part.
func ImagePart(url string) ContentPart {
	return ContentPart{Type: "image", URL: url}
}

// ToolCall is a model-requested tool invocation.
type ToolCall struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// Message is one conversation message.
type Message struct {
	Role       Role          `json:"role"`
	Content    []ContentPart `json:"content,omitempty"`
	ToolCalls  []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
	Name       string        `json:"name,omitempty"`
}

// Text concatenates text content parts.
func (m Message) Text() string {
	text := ""
	for _, part := range m.Content {
		if part.Type != "text" {
			continue
		}
		if text != "" {
			text += "\n"
		}
		text += part.Text
	}
	return text
}

// Usage is token accounting.
type Usage struct {
	InputTokens  uint32 `json:"input_tokens"`
	OutputTokens uint32 `json:"output_tokens"`
}

// ToolSpec describes a tool exposed to the model.
type ToolSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// ModelRequest is passed to a custom model handler.
type ModelRequest struct {
	Messages    []Message       `json:"messages"`
	Tools       []ToolSpec      `json:"tools,omitempty"`
	ToolChoice  json.RawMessage `json:"tool_choice,omitempty"`
	Temperature *float32        `json:"temperature,omitempty"`
	MaxTokens   *uint32         `json:"max_tokens,omitempty"`
}

// ModelResponse is returned by a custom model handler.
type ModelResponse struct {
	Message      Message         `json:"message"`
	FinishReason string          `json:"finish_reason"`
	Usage        Usage           `json:"usage,omitempty"`
	Model        string          `json:"model,omitempty"`
	Raw          json.RawMessage `json:"raw,omitempty"`
}

// ModelDelta is one incremental text or tool-call fragment.
type ModelDelta struct {
	Type      string  `json:"type"`
	Text      string  `json:"text,omitempty"`
	Index     int     `json:"index,omitempty"`
	ID        *string `json:"id,omitempty"`
	Name      *string `json:"name,omitempty"`
	Arguments *string `json:"arguments,omitempty"`
}

// ToolOutput is the result returned to the model.
type ToolOutput struct {
	Content string `json:"content"`
	IsError bool   `json:"is_error,omitempty"`
}

// AgentOutput is the final result of one run.
type AgentOutput struct {
	RunID        string          `json:"run_id"`
	SessionID    string          `json:"session_id"`
	Message      Message         `json:"message"`
	Text         string          `json:"text"`
	Turns        int             `json:"turns"`
	Usage        Usage           `json:"usage"`
	FinishReason json.RawMessage `json:"finish_reason"`
}

// Event is one lifecycle event. Raw retains event-specific fields.
type Event struct {
	Type      string         `json:"type"`
	RunID     string         `json:"run_id"`
	SessionID string         `json:"session_id"`
	Sequence  uint64         `json:"sequence"`
	Turn      int            `json:"turn,omitempty"`
	Delta     *ModelDelta    `json:"delta,omitempty"`
	Raw       map[string]any `json:"-"`
}

func parseEvent(data []byte) (Event, error) {
	var event Event
	if err := json.Unmarshal(data, &event); err != nil {
		return Event{}, err
	}
	if err := json.Unmarshal(data, &event.Raw); err != nil {
		return Event{}, err
	}
	return event, nil
}

// ModelHandler is a non-streaming custom model.
type ModelHandler func(ModelRequest) (ModelResponse, error)

// StreamingModelHandler can emit token deltas while producing a response.
type StreamingModelHandler func(ModelRequest, func(ModelDelta)) (ModelResponse, error)

// ToolHandler executes one tool call.
type ToolHandler func(callID string, arguments map[string]any) (ToolOutput, error)

// EventHandler receives agent lifecycle events.
type EventHandler func(Event)

// SessionStore persists conversation JSON.
type SessionStore interface {
	// Load returns a session JSON and true when found.
	Load(sessionID string) (json.RawMessage, bool, error)
	// Save creates or replaces a session.
	Save(session json.RawMessage) error
	// Delete removes a session.
	Delete(sessionID string) error
}

// MemoryCategory namespaces one memory entry.
type MemoryCategory struct {
	Name    string `json:"name"`
	OwnerID string `json:"owner_id,omitempty"`
}

// MemoryQuery is passed to long-term memory recall.
type MemoryQuery struct {
	Categories []MemoryCategory `json:"categories,omitempty"`
	Text       string           `json:"text"`
	Tags       []string         `json:"tags,omitempty"`
	Limit      int              `json:"limit"`
}

// MemoryEntry is one durable memory item.
type MemoryEntry struct {
	ID          string          `json:"id"`
	Category    *MemoryCategory `json:"category,omitempty"`
	Text        string          `json:"text"`
	Tags        []string        `json:"tags,omitempty"`
	Metadata    map[string]any  `json:"metadata,omitempty"`
	CreatedAtMS uint64          `json:"created_at_ms,omitempty"`
}

// Memory is pluggable long-term memory.
type Memory interface {
	Recall(query MemoryQuery) ([]MemoryEntry, error)
	Remember(entry MemoryEntry) error
}
