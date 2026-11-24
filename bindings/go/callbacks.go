package kestrel

/*
#cgo CFLAGS: -I${SRCDIR}/../../kestrel-ffi/include
#cgo LDFLAGS: -L${SRCDIR}/../../target/debug -lkestrel_ffi
#cgo darwin LDFLAGS: -Wl,-rpath,${SRCDIR}/../../target/debug
#cgo linux LDFLAGS: -Wl,-rpath,${SRCDIR}/../../target/debug
#include <stdlib.h>
#include "kestrel.h"

extern char *kestrelGoStreamModel(
    void *user_data,
    char *request_json,
    KestrelDeltaEmitter emit,
    void *emit_ctx,
    char **error);
extern char *kestrelGoTool(
    void *user_data,
    char *call_id,
    char *arguments_json);
extern void kestrelGoEvent(
    void *user_data,
    char *event_json);
extern char *kestrelGoSessionLoad(
    void *user_data,
    char *session_id,
    char **error);
extern int32_t kestrelGoSessionSave(
    void *user_data,
    char *session_json,
    char **error);
extern int32_t kestrelGoSessionDelete(
    void *user_data,
    char *session_id,
    char **error);
extern char *kestrelGoMemoryRecall(
    void *user_data,
    char *query_json,
    char **error);
extern int32_t kestrelGoMemoryRemember(
    void *user_data,
    char *entry_json,
    char **error);

static char *bridge_stream_fn(
    void *user_data,
    const char *request_json,
    KestrelDeltaEmitter emit,
    void *emit_ctx,
    char **error) {
    return kestrelGoStreamModel(
        user_data, (char *)request_json, emit, emit_ctx, error);
}
static char *bridge_tool_fn(
    void *user_data,
    const char *call_id,
    const char *arguments_json) {
    return kestrelGoTool(
        user_data, (char *)call_id, (char *)arguments_json);
}
static void bridge_event_fn(
    void *user_data,
    const char *event_json) {
    kestrelGoEvent(user_data, (char *)event_json);
}
static char *bridge_session_load_fn(
    void *user_data,
    const char *session_id,
    char **error) {
    return kestrelGoSessionLoad(user_data, (char *)session_id, error);
}
static int32_t bridge_session_save_fn(
    void *user_data,
    const char *session_json,
    char **error) {
    return kestrelGoSessionSave(user_data, (char *)session_json, error);
}
static int32_t bridge_session_delete_fn(
    void *user_data,
    const char *session_id,
    char **error) {
    return kestrelGoSessionDelete(user_data, (char *)session_id, error);
}
static char *bridge_memory_recall_fn(
    void *user_data,
    const char *query_json,
    char **error) {
    return kestrelGoMemoryRecall(user_data, (char *)query_json, error);
}
static int32_t bridge_memory_remember_fn(
    void *user_data,
    const char *entry_json,
    char **error) {
    return kestrelGoMemoryRemember(user_data, (char *)entry_json, error);
}
static KestrelStreamingModelCallback bridge_stream_model(void) {
    return bridge_stream_fn;
}
static KestrelToolCallback bridge_tool(void) {
    return bridge_tool_fn;
}
static KestrelEventCallback bridge_event(void) {
    return bridge_event_fn;
}
static KestrelSessionLoadCallback bridge_session_load(void) {
    return bridge_session_load_fn;
}
static KestrelSessionSaveCallback bridge_session_save(void) {
    return bridge_session_save_fn;
}
static KestrelSessionDeleteCallback bridge_session_delete(void) {
    return bridge_session_delete_fn;
}
static KestrelMemoryRecallCallback bridge_memory_recall(void) {
    return bridge_memory_recall_fn;
}
static KestrelMemoryRememberCallback bridge_memory_remember(void) {
    return bridge_memory_remember_fn;
}
static void invoke_delta(
    KestrelDeltaEmitter emit,
    void *emit_ctx,
    const char *delta_json) {
    emit(emit_ctx, delta_json);
}
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"sync"
	"unsafe"
)

type agentState struct {
	mu           sync.RWMutex
	model        ModelHandler
	streamModel  StreamingModelHandler
	eventHandler EventHandler
}

type toolState struct {
	handler ToolHandler
}

type sessionStoreState struct {
	store SessionStore
}

type memoryCallbackState struct {
	memory Memory
}

var agentStates sync.Map
var toolStates sync.Map
var sessionStoreStates sync.Map
var memoryStates sync.Map

func allocateToken() unsafe.Pointer {
	return C.malloc(1)
}

func freeToken(token unsafe.Pointer) {
	if token != nil {
		C.free(token)
	}
}

func allocateKestrelString(value string) *C.char {
	valueC := C.CString(value)
	defer C.free(unsafe.Pointer(valueC))
	return C.kestrel_string_new(valueC)
}

func setCallbackError(errorOut **C.char, err error) {
	if errorOut != nil {
		*errorOut = allocateKestrelString(err.Error())
	}
}

func streamModelCallback() C.KestrelStreamingModelCallback {
	return C.bridge_stream_model()
}

func toolCallback() C.KestrelToolCallback {
	return C.bridge_tool()
}

func eventCallback() C.KestrelEventCallback {
	return C.bridge_event()
}

func sessionLoadCallback() C.KestrelSessionLoadCallback {
	return C.bridge_session_load()
}

func sessionSaveCallback() C.KestrelSessionSaveCallback {
	return C.bridge_session_save()
}

func sessionDeleteCallback() C.KestrelSessionDeleteCallback {
	return C.bridge_session_delete()
}

func memoryRecallCallback() C.KestrelMemoryRecallCallback {
	return C.bridge_memory_recall()
}

func memoryRememberCallback() C.KestrelMemoryRememberCallback {
	return C.bridge_memory_remember()
}

func emitModelDelta(
	emit C.KestrelDeltaEmitter,
	emitCtx unsafe.Pointer,
	delta ModelDelta,
) {
	data, err := json.Marshal(delta)
	if err != nil {
		return
	}
	dataC := C.CString(string(data))
	defer C.free(unsafe.Pointer(dataC))
	C.invoke_delta(emit, emitCtx, dataC)
}

//export kestrelGoStreamModel
func kestrelGoStreamModel(
	userData unsafe.Pointer,
	requestJSON *C.char,
	emit C.KestrelDeltaEmitter,
	emitCtx unsafe.Pointer,
	errorOut **C.char,
) (result *C.char) {
	defer func() {
		if recovered := recover(); recovered != nil {
			setCallbackError(errorOut, fmt.Errorf("Go model callback panic: %v", recovered))
			result = nil
		}
	}()

	value, ok := agentStates.Load(uintptr(userData))
	if !ok {
		setCallbackError(errorOut, fmt.Errorf("unknown Go agent callback"))
		return nil
	}
	state := value.(*agentState)
	if requestJSON == nil {
		setCallbackError(errorOut, fmt.Errorf("model request is NULL"))
		return nil
	}

	var request ModelRequest
	if err := json.Unmarshal([]byte(C.GoString(requestJSON)), &request); err != nil {
		setCallbackError(errorOut, fmt.Errorf("decode model request: %w", err))
		return nil
	}

	state.mu.RLock()
	streamHandler := state.streamModel
	handler := state.model
	state.mu.RUnlock()

	var response ModelResponse
	var err error
	switch {
	case streamHandler != nil:
		response, err = streamHandler(request, func(delta ModelDelta) {
			emitModelDelta(emit, emitCtx, delta)
		})
	case handler != nil:
		response, err = handler(request)
		if text := response.Message.Text(); text != "" {
			emitModelDelta(emit, emitCtx, ModelDelta{Type: "text", Text: text})
		}
	default:
		err = fmt.Errorf("no Go model handler is configured")
	}
	if err != nil {
		setCallbackError(errorOut, err)
		return nil
	}
	data, err := json.Marshal(response)
	if err != nil {
		setCallbackError(errorOut, fmt.Errorf("encode model response: %w", err))
		return nil
	}
	return allocateKestrelString(string(data))
}

//export kestrelGoTool
func kestrelGoTool(
	userData unsafe.Pointer,
	callID *C.char,
	argumentsJSON *C.char,
) (result *C.char) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = allocateKestrelString(mustJSON(ToolOutput{
				Content: fmt.Sprintf("Go tool callback panic: %v", recovered),
				IsError: true,
			}))
		}
	}()

	value, ok := toolStates.Load(uintptr(userData))
	if !ok {
		return allocateKestrelString(mustJSON(ToolOutput{
			Content: "unknown Go tool callback",
			IsError: true,
		}))
	}
	tool := value.(*toolState)
	arguments := map[string]any{}
	if argumentsJSON != nil {
		if err := json.Unmarshal([]byte(C.GoString(argumentsJSON)), &arguments); err != nil {
			return allocateKestrelString(mustJSON(ToolOutput{
				Content: fmt.Sprintf("decode tool arguments: %v", err),
				IsError: true,
			}))
		}
	}
	output, err := tool.handler(C.GoString(callID), arguments)
	if err != nil {
		return allocateKestrelString(mustJSON(ToolOutput{
			Content: err.Error(),
			IsError: true,
		}))
	}
	return allocateKestrelString(mustJSON(output))
}

//export kestrelGoEvent
func kestrelGoEvent(userData unsafe.Pointer, eventJSON *C.char) {
	defer func() {
		_ = recover()
	}()
	value, ok := agentStates.Load(uintptr(userData))
	if !ok || eventJSON == nil {
		return
	}
	state := value.(*agentState)
	state.mu.RLock()
	handler := state.eventHandler
	state.mu.RUnlock()
	if handler == nil {
		return
	}
	event, err := parseEvent([]byte(C.GoString(eventJSON)))
	if err != nil {
		return
	}
	handler(event)
}

//export kestrelGoSessionLoad
func kestrelGoSessionLoad(
	userData unsafe.Pointer,
	sessionID *C.char,
	errorOut **C.char,
) (result *C.char) {
	defer func() {
		if recovered := recover(); recovered != nil {
			setCallbackError(errorOut, fmt.Errorf("Go session load panic: %v", recovered))
			result = nil
		}
	}()
	value, ok := sessionStoreStates.Load(uintptr(userData))
	if !ok {
		setCallbackError(errorOut, fmt.Errorf("unknown Go session store callback"))
		return nil
	}
	session, found, err := value.(*sessionStoreState).store.Load(C.GoString(sessionID))
	if err != nil {
		setCallbackError(errorOut, err)
		return nil
	}
	if !found {
		return nil
	}
	return allocateKestrelString(string(session))
}

//export kestrelGoSessionSave
func kestrelGoSessionSave(
	userData unsafe.Pointer,
	sessionJSON *C.char,
	errorOut **C.char,
) C.int32_t {
	defer func() {
		if recovered := recover(); recovered != nil {
			setCallbackError(errorOut, fmt.Errorf("Go session save panic: %v", recovered))
		}
	}()
	value, ok := sessionStoreStates.Load(uintptr(userData))
	if !ok {
		setCallbackError(errorOut, fmt.Errorf("unknown Go session store callback"))
		return -1
	}
	if err := value.(*sessionStoreState).store.Save(json.RawMessage(C.GoString(sessionJSON))); err != nil {
		setCallbackError(errorOut, err)
		return -1
	}
	return 0
}

//export kestrelGoSessionDelete
func kestrelGoSessionDelete(
	userData unsafe.Pointer,
	sessionID *C.char,
	errorOut **C.char,
) C.int32_t {
	defer func() {
		if recovered := recover(); recovered != nil {
			setCallbackError(errorOut, fmt.Errorf("Go session delete panic: %v", recovered))
		}
	}()
	value, ok := sessionStoreStates.Load(uintptr(userData))
	if !ok {
		setCallbackError(errorOut, fmt.Errorf("unknown Go session store callback"))
		return -1
	}
	if err := value.(*sessionStoreState).store.Delete(C.GoString(sessionID)); err != nil {
		setCallbackError(errorOut, err)
		return -1
	}
	return 0
}

//export kestrelGoMemoryRecall
func kestrelGoMemoryRecall(
	userData unsafe.Pointer,
	queryJSON *C.char,
	errorOut **C.char,
) (result *C.char) {
	defer func() {
		if recovered := recover(); recovered != nil {
			setCallbackError(errorOut, fmt.Errorf("Go memory recall panic: %v", recovered))
			result = nil
		}
	}()
	value, ok := memoryStates.Load(uintptr(userData))
	if !ok {
		setCallbackError(errorOut, fmt.Errorf("unknown Go memory callback"))
		return nil
	}
	var query MemoryQuery
	if err := json.Unmarshal([]byte(C.GoString(queryJSON)), &query); err != nil {
		setCallbackError(errorOut, fmt.Errorf("decode memory query: %w", err))
		return nil
	}
	entries, err := value.(*memoryCallbackState).memory.Recall(query)
	if err != nil {
		setCallbackError(errorOut, err)
		return nil
	}
	if entries == nil {
		entries = []MemoryEntry{}
	}
	return allocateKestrelString(mustJSON(entries))
}

//export kestrelGoMemoryRemember
func kestrelGoMemoryRemember(
	userData unsafe.Pointer,
	entryJSON *C.char,
	errorOut **C.char,
) C.int32_t {
	defer func() {
		if recovered := recover(); recovered != nil {
			setCallbackError(errorOut, fmt.Errorf("Go memory remember panic: %v", recovered))
		}
	}()
	value, ok := memoryStates.Load(uintptr(userData))
	if !ok {
		setCallbackError(errorOut, fmt.Errorf("unknown Go memory callback"))
		return -1
	}
	var entry MemoryEntry
	if err := json.Unmarshal([]byte(C.GoString(entryJSON)), &entry); err != nil {
		setCallbackError(errorOut, fmt.Errorf("decode memory entry: %w", err))
		return -1
	}
	if err := value.(*memoryCallbackState).memory.Remember(entry); err != nil {
		setCallbackError(errorOut, err)
		return -1
	}
	return 0
}

func mustJSON(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return `{"content":"failed to encode Go callback result","is_error":true}`
	}
	return string(data)
}
