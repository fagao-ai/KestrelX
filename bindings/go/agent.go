package kestrel

/*
#include <stdlib.h>
#include "kestrel.h"
*/
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// Error is returned by the native Kestrel library.
type Error struct {
	Message string
}

func (e *Error) Error() string {
	return e.Message
}

// Option configures an Agent.
type Option func(*agentOptions) error

type agentOptions struct {
	model        ModelHandler
	streamModel  StreamingModelHandler
	eventHandler EventHandler
	sessionStore SessionStore
	memory       Memory
}

// WithModel installs a non-streaming custom model.
func WithModel(handler ModelHandler) Option {
	return func(options *agentOptions) error {
		options.model = handler
		return nil
	}
}

// WithStreamingModel installs a model that can emit token deltas.
func WithStreamingModel(handler StreamingModelHandler) Option {
	return func(options *agentOptions) error {
		options.streamModel = handler
		return nil
	}
}

// WithEventHandler installs a lifecycle event handler.
func WithEventHandler(handler EventHandler) Option {
	return func(options *agentOptions) error {
		options.eventHandler = handler
		return nil
	}
}

// WithSessionStore installs host-backed conversation persistence.
func WithSessionStore(store SessionStore) Option {
	return func(options *agentOptions) error {
		options.sessionStore = store
		return nil
	}
}

// WithMemory installs host-backed long-term memory.
func WithMemory(memory Memory) Option {
	return func(options *agentOptions) error {
		options.memory = memory
		return nil
	}
}

// Agent owns a native Kestrel agent.
type Agent struct {
	mu          sync.Mutex
	ptr         *C.KestrelAgent
	token       unsafe.Pointer
	storeToken  unsafe.Pointer
	memoryToken unsafe.Pointer
	toolTokens  []unsafe.Pointer
	state       *agentState
	runs        map[*Run]struct{}
	closed      bool
}

// New creates an agent from configuration JSON.
//
// config may be a JSON string, []byte, json.RawMessage, or any JSON-marshalable
// value. If no model option is supplied, config.openai is used.
func New(config any, options ...Option) (*Agent, error) {
	var opts agentOptions
	for _, option := range options {
		if err := option(&opts); err != nil {
			return nil, err
		}
	}
	if opts.model != nil && opts.streamModel != nil {
		return nil, errors.New("WithModel and WithStreamingModel are mutually exclusive")
	}
	configJSON, err := encodeConfig(config)
	if err != nil {
		return nil, err
	}
	configC := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(configC))

	state := &agentState{
		model:        opts.model,
		streamModel:  opts.streamModel,
		eventHandler: opts.eventHandler,
	}
	token := allocateToken()
	agentStates.Store(uintptr(token), state)

	var native *C.KestrelAgent
	var errorC *C.char
	hasModel := opts.model != nil || opts.streamModel != nil
	var storeToken unsafe.Pointer
	var memoryToken unsafe.Pointer
	cleanup := func(native *C.KestrelAgent) {
		if native != nil {
			C.kestrel_agent_free(native)
		}
		if storeToken != nil {
			sessionStoreStates.Delete(uintptr(storeToken))
			freeToken(storeToken)
		}
		if memoryToken != nil {
			memoryStates.Delete(uintptr(memoryToken))
			freeToken(memoryToken)
		}
		agentStates.Delete(uintptr(token))
		freeToken(token)
	}

	if opts.sessionStore != nil || opts.memory != nil {
		var builder *C.KestrelAgentBuilder
		if hasModel {
			builder = C.kestrel_agent_builder_new_with_streaming_model(
				configC,
				streamModelCallback(),
				token,
				&errorC,
			)
		} else {
			builder = C.kestrel_agent_builder_new(configC, &errorC)
		}
		if builder == nil {
			cleanup(nil)
			return nil, takeCError(errorC)
		}
		if opts.sessionStore != nil {
			storeToken = allocateToken()
			sessionStoreStates.Store(uintptr(storeToken), &sessionStoreState{store: opts.sessionStore})
			if status := C.kestrel_agent_builder_set_session_store(
				builder,
				sessionLoadCallback(),
				sessionSaveCallback(),
				sessionDeleteCallback(),
				storeToken,
				&errorC,
			); status != 0 {
				C.kestrel_agent_builder_free(builder)
				cleanup(nil)
				return nil, takeCError(errorC)
			}
		}
		if opts.memory != nil {
			memoryToken = allocateToken()
			memoryStates.Store(uintptr(memoryToken), &memoryCallbackState{memory: opts.memory})
			if status := C.kestrel_agent_builder_set_memory(
				builder,
				memoryRecallCallback(),
				memoryRememberCallback(),
				memoryToken,
				&errorC,
			); status != 0 {
				C.kestrel_agent_builder_free(builder)
				cleanup(nil)
				return nil, takeCError(errorC)
			}
		}
		native = C.kestrel_agent_builder_build(builder, &errorC)
		C.kestrel_agent_builder_free(builder)
	} else if hasModel {
		native = C.kestrel_agent_new_with_streaming_model(
			configC,
			streamModelCallback(),
			token,
			&errorC,
		)
	} else {
		native = C.kestrel_agent_new(configC, &errorC)
	}
	if native == nil {
		cleanup(nil)
		return nil, takeCError(errorC)
	}
	if status := C.kestrel_agent_set_event_callback(native, eventCallback(), token); status != 0 {
		cleanup(native)
		return nil, errors.New("failed to install Go event callback")
	}

	return &Agent{
		ptr:         native,
		token:       token,
		storeToken:  storeToken,
		memoryToken: memoryToken,
		state:       state,
		runs:        make(map[*Run]struct{}),
	}, nil
}

// SetEventHandler replaces the lifecycle event handler.
func (a *Agent) SetEventHandler(handler EventHandler) {
	a.state.mu.Lock()
	a.state.eventHandler = handler
	a.state.mu.Unlock()
}

// RegisterTool adds one synchronous Go tool.
func (a *Agent) RegisterTool(spec ToolSpec, handler ToolHandler) error {
	if handler == nil {
		return errors.New("tool handler must not be nil")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return errors.New("agent is closed")
	}
	schema, err := json.Marshal(spec.InputSchema)
	if err != nil {
		return fmt.Errorf("encode tool schema: %w", err)
	}
	nameC := C.CString(spec.Name)
	defer C.free(unsafe.Pointer(nameC))
	descriptionC := C.CString(spec.Description)
	defer C.free(unsafe.Pointer(descriptionC))
	schemaC := C.CString(string(schema))
	defer C.free(unsafe.Pointer(schemaC))

	token := allocateToken()
	toolStates.Store(uintptr(token), &toolState{handler: handler})
	var errorC *C.char
	status := C.kestrel_agent_register_tool(
		a.ptr,
		nameC,
		descriptionC,
		schemaC,
		toolCallback(),
		token,
		&errorC,
	)
	if status != 0 {
		toolStates.Delete(uintptr(token))
		freeToken(token)
		return takeCError(errorC)
	}

	a.toolTokens = append(a.toolTokens, token)
	return nil
}

// Run executes one prompt and waits for completion.
func (a *Agent) Run(ctx context.Context, prompt string) (*AgentOutput, error) {
	run, err := a.StartRun(prompt)
	if err != nil {
		return nil, err
	}
	defer run.Close()
	return run.Wait(ctx)
}

// RunMessage executes one typed user message and waits for completion.
func (a *Agent) RunMessage(ctx context.Context, message Message) (*AgentOutput, error) {
	run, err := a.StartRunMessage(message)
	if err != nil {
		return nil, err
	}
	defer run.Close()
	return run.Wait(ctx)
}

// StartRun starts a nonblocking run.
func (a *Agent) StartRun(prompt string) (*Run, error) {
	promptC := C.CString(prompt)
	defer C.free(unsafe.Pointer(promptC))
	return a.startRunC(func() (*C.KestrelRun, *C.char) {
		var errorC *C.char
		native := C.kestrel_agent_start_run(a.ptr, promptC, &errorC)
		return native, errorC
	})
}

// StartRunMessage starts a nonblocking run from a typed message.
func (a *Agent) StartRunMessage(message Message) (*Run, error) {
	data, err := json.Marshal(message)
	if err != nil {
		return nil, fmt.Errorf("encode message: %w", err)
	}
	messageC := C.CString(string(data))
	defer C.free(unsafe.Pointer(messageC))
	return a.startRunC(func() (*C.KestrelRun, *C.char) {
		var errorC *C.char
		native := C.kestrel_agent_start_run_message(a.ptr, messageC, &errorC)
		return native, errorC
	})
}

func (a *Agent) startRunC(start func() (*C.KestrelRun, *C.char)) (*Run, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, errors.New("agent is closed")
	}
	native, errorC := start()
	if native == nil {
		return nil, takeCError(errorC)
	}
	run := &Run{owner: a, ptr: native}
	a.runs[run] = struct{}{}
	return run, nil
}

// Cancel requests cancellation of the active run.
func (a *Agent) Cancel() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if !a.closed && a.ptr != nil {
		C.kestrel_agent_cancel(a.ptr)
	}
}

// Session returns the current session JSON.
func (a *Agent) Session() (json.RawMessage, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, errors.New("agent is closed")
	}
	var errorC *C.char
	value := C.kestrel_agent_session_json(a.ptr, &errorC)
	if value == nil {
		return nil, takeCError(errorC)
	}
	defer C.kestrel_string_free(value)
	return json.RawMessage(C.GoString(value)), nil
}

// Todos returns todo JSON. The boolean is false when todos are disabled.
func (a *Agent) Todos() (json.RawMessage, bool, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, false, errors.New("agent is closed")
	}
	var errorC *C.char
	value := C.kestrel_agent_todos_json(a.ptr, &errorC)
	if value == nil {
		if errorC != nil {
			return nil, false, takeCError(errorC)
		}
		return nil, false, nil
	}
	defer C.kestrel_string_free(value)
	return json.RawMessage(C.GoString(value)), true, nil
}

// Reset clears history while retaining the session ID.
func (a *Agent) Reset() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return errors.New("agent is closed")
	}
	var errorC *C.char
	if status := C.kestrel_agent_reset(a.ptr, &errorC); status != 0 {
		return takeCError(errorC)
	}
	return nil
}

// DeleteSession deletes the stored session and clears local history.
func (a *Agent) DeleteSession() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return errors.New("agent is closed")
	}
	var errorC *C.char
	if status := C.kestrel_agent_delete_session(a.ptr, &errorC); status != 0 {
		return takeCError(errorC)
	}
	return nil
}

// Close waits for active runs and releases native resources.
func (a *Agent) Close() {
	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return
	}
	a.closed = true
	runs := make([]*Run, 0, len(a.runs))
	for run := range a.runs {
		runs = append(runs, run)
	}
	toolTokens := a.toolTokens
	a.toolTokens = nil
	storeToken := a.storeToken
	a.storeToken = nil
	memoryToken := a.memoryToken
	a.memoryToken = nil
	native := a.ptr
	a.ptr = nil
	a.mu.Unlock()

	for _, run := range runs {
		run.Close()
	}
	if native != nil {
		C.kestrel_agent_free(native)
	}
	agentStates.Delete(uintptr(a.token))
	freeToken(a.token)
	if storeToken != nil {
		sessionStoreStates.Delete(uintptr(storeToken))
		freeToken(storeToken)
	}
	if memoryToken != nil {
		memoryStates.Delete(uintptr(memoryToken))
		freeToken(memoryToken)
	}
	for _, token := range toolTokens {
		toolStates.Delete(uintptr(token))
		freeToken(token)
	}
}

func (a *Agent) removeRun(run *Run) {
	a.mu.Lock()
	delete(a.runs, run)
	a.mu.Unlock()
}

func encodeConfig(config any) ([]byte, error) {
	switch value := config.(type) {
	case string:
		return []byte(value), nil
	case []byte:
		return value, nil
	case json.RawMessage:
		return value, nil
	default:
		data, err := json.Marshal(config)
		if err != nil {
			return nil, fmt.Errorf("encode config: %w", err)
		}
		return data, nil
	}
}

func takeCError(value *C.char) error {
	if value == nil {
		return errors.New("unknown Kestrel error")
	}
	defer C.kestrel_string_free(value)
	return &Error{Message: C.GoString(value)}
}
