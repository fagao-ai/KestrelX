package kestrel

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

type mapSessionStore struct {
	mu       sync.Mutex
	sessions map[string]json.RawMessage
}

func (s *mapSessionStore) Load(id string) (json.RawMessage, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	value, ok := s.sessions[id]
	return append(json.RawMessage(nil), value...), ok, nil
}

func (s *mapSessionStore) Save(value json.RawMessage) error {
	var session struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(value, &session); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sessions == nil {
		s.sessions = make(map[string]json.RawMessage)
	}
	s.sessions[session.ID] = append(json.RawMessage(nil), value...)
	return nil
}

func (s *mapSessionStore) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, id)
	return nil
}

type sliceMemory struct {
	mu      sync.Mutex
	entries []MemoryEntry
}

func (m *sliceMemory) Recall(MemoryQuery) ([]MemoryEntry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]MemoryEntry(nil), m.entries...), nil
}

func (m *sliceMemory) Remember(entry MemoryEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append(m.entries, entry)
	return nil
}

func TestAgentToolAndStreaming(t *testing.T) {
	var mu sync.Mutex
	modelCalls := 0
	toolCalls := 0
	var eventDeltas []string

	model := func(request ModelRequest, emit func(ModelDelta)) (ModelResponse, error) {
		mu.Lock()
		modelCalls++
		call := modelCalls
		mu.Unlock()
		if call == 1 {
			emit(ModelDelta{Type: "text", Text: "calling tool"})
			return ModelResponse{
				Message: Message{
					Role: RoleAssistant,
					ToolCalls: []ToolCall{{
						ID:   "call-1",
						Name: "echo",
						Arguments: map[string]any{
							"text": "hello",
						},
					}},
				},
				FinishReason: "tool_calls",
			}, nil
		}
		toolResult := ""
		for _, message := range request.Messages {
			if message.Role == RoleTool {
				toolResult = message.Text()
			}
		}
		text := "final: " + toolResult
		emit(ModelDelta{Type: "text", Text: text})
		return ModelResponse{
			Message: Message{
				Role:    RoleAssistant,
				Content: []ContentPart{TextPart(text)},
			},
			FinishReason: "stop",
		}, nil
	}

	agent, err := New(
		map[string]any{"streaming": true},
		WithStreamingModel(model),
		WithEventHandler(func(event Event) {
			if event.Type == "model_delta" && event.Delta != nil {
				mu.Lock()
				eventDeltas = append(eventDeltas, event.Delta.Text)
				mu.Unlock()
			}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Close()

	err = agent.RegisterTool(
		ToolSpec{
			Name:        "echo",
			Description: "Echo text",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
				},
				"required": []string{"text"},
			},
		},
		func(_ string, arguments map[string]any) (ToolOutput, error) {
			mu.Lock()
			toolCalls++
			mu.Unlock()
			return ToolOutput{Content: "echo:" + arguments["text"].(string)}, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	output, err := agent.Run(ctx, "use the tool")
	if err != nil {
		t.Fatal(err)
	}
	if output.Text != "final: echo:hello" {
		t.Fatalf("unexpected output: %q", output.Text)
	}
	mu.Lock()
	if toolCalls != 1 {
		t.Fatalf("tool calls = %d", toolCalls)
	}
	if strings.Join(eventDeltas, "|") != "calling tool|final: echo:hello" {
		t.Fatalf("unexpected deltas: %#v", eventDeltas)
	}
	mu.Unlock()

	session, err := agent.Session()
	if err != nil {
		t.Fatal(err)
	}
	var sessionValue map[string]any
	if err := json.Unmarshal(session, &sessionValue); err != nil {
		t.Fatal(err)
	}
	if len(sessionValue["messages"].([]any)) != 4 {
		t.Fatalf("unexpected session: %s", session)
	}
}

func TestStartRunAndNextEvent(t *testing.T) {
	var mu sync.Mutex
	modelCalls := 0
	agent, err := New(
		map[string]any{"streaming": true},
		WithStreamingModel(func(_ ModelRequest, emit func(ModelDelta)) (ModelResponse, error) {
			mu.Lock()
			modelCalls++
			mu.Unlock()
			emit(ModelDelta{Type: "text", Text: "async"})
			return ModelResponse{
				Message: Message{
					Role:    RoleAssistant,
					Content: []ContentPart{TextPart("async answer")},
				},
				FinishReason: "stop",
			}, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	run, err := agent.StartRun("async prompt")
	if err != nil {
		t.Fatal(err)
	}
	defer run.Close()

	modelDeltas := 0
	for {
		event, err := run.NextEvent(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if event.Type == "model_delta" && event.Delta != nil &&
			event.Delta.Text == "async" {
			modelDeltas++
		}
	}
	output, err := run.Wait(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if output.Text != "async answer" || modelDeltas != 1 {
		t.Fatalf("output=%q deltas=%d", output.Text, modelDeltas)
	}
	mu.Lock()
	defer mu.Unlock()
	if modelCalls != 1 {
		t.Fatalf("model calls = %d", modelCalls)
	}
}

func TestGoSessionStoreAndMemory(t *testing.T) {
	store := &mapSessionStore{}
	memory := &sliceMemory{}
	var mu sync.Mutex
	modelCalls := 0
	var requests []ModelRequest

	model := func(request ModelRequest) (ModelResponse, error) {
		mu.Lock()
		modelCalls++
		call := modelCalls
		requests = append(requests, request)
		mu.Unlock()
		if call == 1 {
			return ModelResponse{
				Message: Message{
					Role: RoleAssistant,
					ToolCalls: []ToolCall{{
						ID:   "remember-1",
						Name: "memory_remember",
						Arguments: map[string]any{
							"text": "The user lives in Shanghai",
						},
					}},
				},
				FinishReason: "tool_calls",
			}, nil
		}
		text := "first answer"
		if call > 2 {
			text = "second answer"
		}
		return ModelResponse{
			Message: Message{
				Role:    RoleAssistant,
				Content: []ContentPart{TextPart(text)},
			},
			FinishReason: "stop",
		}, nil
	}

	config := map[string]any{
		"session_id": "go-persisted-session",
		"streaming":  true,
		"openai":     map[string]any{"model": "go-callback"},
		"memory":     map[string]any{"enabled": true, "tools": true},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	first, err := New(config, WithModel(model), WithSessionStore(store), WithMemory(memory))
	if err != nil {
		t.Fatal(err)
	}
	firstOutput, err := first.Run(ctx, "first prompt")
	if err != nil {
		first.Close()
		t.Fatal(err)
	}
	if firstOutput.Text != "first answer" {
		t.Fatalf("unexpected first output: %q", firstOutput.Text)
	}
	first.Close()

	second, err := New(config, WithModel(model), WithSessionStore(store), WithMemory(memory))
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	secondOutput, err := second.Run(ctx, "second prompt")
	if err != nil {
		t.Fatal(err)
	}
	if secondOutput.Text != "second answer" {
		t.Fatalf("unexpected second output: %q", secondOutput.Text)
	}
	if err := second.DeleteSession(); err != nil {
		t.Fatal(err)
	}
	store.mu.Lock()
	if len(store.sessions) != 0 {
		t.Fatalf("session store was not cleared: %#v", store.sessions)
	}
	store.mu.Unlock()

	mu.Lock()
	defer mu.Unlock()
	history := mustJSON(requests[len(requests)-1])
	if !strings.Contains(history, "first prompt") ||
		!strings.Contains(history, "first answer") ||
		!strings.Contains(history, "second prompt") ||
		!strings.Contains(history, "Shanghai") {
		t.Fatalf("persistence missing from request: %s", history)
	}
	memory.mu.Lock()
	defer memory.mu.Unlock()
	if len(memory.entries) != 1 {
		t.Fatalf("memory entries = %d", len(memory.entries))
	}
}

func TestResumeAndStartResume(t *testing.T) {
	store := &mapSessionStore{}
	modelCalls := 0
	model := func(ModelRequest) (ModelResponse, error) {
		modelCalls++
		return ModelResponse{
			Message: Message{
				Role:    RoleAssistant,
				Content: []ContentPart{TextPart(fmt.Sprintf("resumed %d", modelCalls))},
			},
			FinishReason: "stop",
		}, nil
	}
	config := map[string]any{
		"session_id": "go-resume",
		"streaming":  false,
		"openai":     map[string]any{"model": "go-resume-model"},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	seed := func(runID string) {
		t.Helper()
		value, err := json.Marshal(map[string]any{
			"schema_version": 1,
			"id":             "go-resume",
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "finish the run"},
					},
				},
			},
			"metadata":      map[string]any{},
			"created_at_ms": 1,
			"updated_at_ms": 1,
			"revision":      1,
			"active_run": map[string]any{
				"run_id":        runID,
				"turn":          1,
				"usage":         map[string]any{"input_tokens": 0, "output_tokens": 0},
				"phase":         map[string]any{"type": "model_call"},
				"updated_at_ms": 1,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		store.mu.Lock()
		if store.sessions == nil {
			store.sessions = make(map[string]json.RawMessage)
		}
		store.sessions["go-resume"] = value
		store.mu.Unlock()
	}

	seed("go-run-1")
	agent, err := New(config, WithModel(model), WithSessionStore(store))
	if err != nil {
		t.Fatal(err)
	}
	output, err := agent.Resume(ctx, false)
	if err != nil {
		agent.Close()
		t.Fatal(err)
	}
	if output.RunID != "go-run-1" || output.Text != "resumed 1" {
		t.Fatalf("unexpected resume output: %#v", output)
	}
	agent.Close()

	seed("go-run-2")
	agent, err = New(config, WithModel(model), WithSessionStore(store))
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Close()
	run, err := agent.StartResume(false)
	if err != nil {
		t.Fatal(err)
	}
	defer run.Close()
	output, err = run.Wait(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if output.RunID != "go-run-2" || output.Text != "resumed 2" {
		t.Fatalf("unexpected async resume output: %#v", output)
	}
}
