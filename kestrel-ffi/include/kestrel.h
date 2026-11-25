#ifndef KESTREL_H
#define KESTREL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KestrelAgent KestrelAgent;
typedef struct KestrelAgentBuilder KestrelAgentBuilder;
typedef struct KestrelRun KestrelRun;

enum {
    KESTREL_STATUS_ERROR = -1,
    KESTREL_STATUS_TIMEOUT = 0,
    KESTREL_STATUS_OK = 1,
    KESTREL_STATUS_EVENT_END = 2
};

/*
 * Event JSON is borrowed for the duration of the callback. Copy it before
 * returning. Callbacks must be fast and must not throw or unwind.
 */
typedef void (*KestrelEventCallback)(void *user_data, const char *event_json);

/*
 * Return a UTF-8 string allocated with kestrel_string_new. It may be either a
 * ToolOutput JSON object such as {"content":"ok","is_error":false}, or plain
 * text. Kestrel takes ownership of the returned string.
 */
typedef char *(*KestrelToolCallback)(
    void *user_data,
    const char *call_id,
    const char *arguments_json);

/*
 * Request JSON deserializes into kestrel::ModelRequest. Return JSON for
 * kestrel::ModelResponse, allocated with kestrel_string_new. Kestrel takes
 * ownership of the returned string.
 */
typedef char *(*KestrelModelCallback)(
    void *user_data,
    const char *request_json);

/*
 * delta_json deserializes into kestrel::ModelDelta. emit and emit_ctx are
 * borrowed for the duration of the streaming model callback and must not be
 * stored after it returns.
 */
typedef void (*KestrelDeltaEmitter)(
    void *emit_ctx,
    const char *delta_json);

/*
 * Emit zero or more ModelDelta JSON values during generation, then return a
 * complete ModelResponse JSON allocated with kestrel_string_new.
 */
typedef char *(*KestrelStreamingModelCallback)(
    void *user_data,
    const char *request_json,
    KestrelDeltaEmitter emit,
    void *emit_ctx,
    char **error);

/*
 * Persistence callbacks receive error as a char** initialized to NULL.
 * On failure allocate an error with kestrel_string_new and store it in *error.
 * Returned load/recall strings are owned by Kestrel and must also be allocated
 * with kestrel_string_new.
 */
typedef char *(*KestrelSessionLoadCallback)(
    void *user_data,
    const char *session_id,
    char **error);

typedef int32_t (*KestrelSessionSaveCallback)(
    void *user_data,
    const char *session_json,
    char **error);

typedef int32_t (*KestrelSessionDeleteCallback)(
    void *user_data,
    const char *session_id,
    char **error);

typedef char *(*KestrelMemoryRecallCallback)(
    void *user_data,
    const char *query_json,
    char **error);

typedef int32_t (*KestrelMemoryRememberCallback)(
    void *user_data,
    const char *entry_json,
    char **error);

const char *kestrel_version(void);
uint32_t kestrel_contract_version(void);

/* Returns a JSON array of capability names supported by this FFI build. */
char *kestrel_capabilities_json(char **error);

/* Returns 0 when config JSON is valid for this FFI build. */
int32_t kestrel_validate_config(
    const char *config_json,
    int32_t has_custom_model,
    char **error);

/* Allocates a UTF-8 C string with Kestrel's allocator. */
char *kestrel_string_new(const char *value);

/* Frees a string returned by Kestrel or allocated with kestrel_string_new. */
void kestrel_string_free(char *value);

/*
 * Builds an OpenAI-compatible agent from config JSON.
 * On failure returns NULL and, when error is non-NULL, stores an allocated
 * error string in *error that must be freed with kestrel_string_free.
 */
KestrelAgent *kestrel_agent_new(const char *config_json, char **error);

/*
 * Builds an agent whose model is implemented by the host callback.
 * The callback user_data must outlive the agent.
 */
KestrelAgent *kestrel_agent_new_with_model(
    const char *config_json,
    KestrelModelCallback callback,
    void *user_data,
    char **error);

KestrelAgent *kestrel_agent_new_with_streaming_model(
    const char *config_json,
    KestrelStreamingModelCallback callback,
    void *user_data,
    char **error);

/*
 * Flexible construction path. Register persistence callbacks before build.
 * The builder remains owned by the caller and can produce one agent.
 */
KestrelAgentBuilder *kestrel_agent_builder_new(
    const char *config_json,
    char **error);

KestrelAgentBuilder *kestrel_agent_builder_new_with_model(
    const char *config_json,
    KestrelModelCallback callback,
    void *user_data,
    char **error);

KestrelAgentBuilder *kestrel_agent_builder_new_with_streaming_model(
    const char *config_json,
    KestrelStreamingModelCallback callback,
    void *user_data,
    char **error);

int32_t kestrel_agent_builder_set_session_store(
    KestrelAgentBuilder *builder,
    KestrelSessionLoadCallback load,
    KestrelSessionSaveCallback save,
    KestrelSessionDeleteCallback delete_callback,
    void *user_data,
    char **error);

int32_t kestrel_agent_builder_set_memory(
    KestrelAgentBuilder *builder,
    KestrelMemoryRecallCallback recall,
    KestrelMemoryRememberCallback remember,
    void *user_data,
    char **error);

KestrelAgent *kestrel_agent_builder_build(
    KestrelAgentBuilder *builder,
    char **error);

void kestrel_agent_builder_free(KestrelAgentBuilder *builder);

void kestrel_agent_free(KestrelAgent *agent);

int32_t kestrel_agent_set_event_callback(
    KestrelAgent *agent,
    KestrelEventCallback callback,
    void *user_data);

int32_t kestrel_agent_register_tool(
    KestrelAgent *agent,
    const char *name,
    const char *description,
    const char *input_schema_json,
    KestrelToolCallback callback,
    void *user_data,
    char **error);

/* Runs synchronously. Event callbacks execute on an internal runtime thread. */
char *kestrel_agent_run(
    KestrelAgent *agent,
    const char *prompt,
    char **error);

/* Accepts a kestrel::Message JSON object, for multimodal/user messages. */
char *kestrel_agent_run_message(
    KestrelAgent *agent,
    const char *message_json,
    char **error);

/*
 * Resumes an interrupted run. retry_uncertain_tools non-zero permits re-running
 * a tool whose result was not checkpointed.
 */
char *kestrel_agent_resume(
    KestrelAgent *agent,
    int32_t retry_uncertain_tools,
    char **error);

/* Starts a nonblocking run. Event callbacks still receive events. */
KestrelRun *kestrel_agent_start_run(
    KestrelAgent *agent,
    const char *prompt,
    char **error);

KestrelRun *kestrel_agent_start_run_message(
    KestrelAgent *agent,
    const char *message_json,
    char **error);

/* Starts a nonblocking resume from the persisted checkpoint. */
KestrelRun *kestrel_agent_start_resume(
    KestrelAgent *agent,
    int32_t retry_uncertain_tools,
    char **error);

/*
 * Returns 1 with event_json, 0 on timeout, 2 when the run has ended, or -1 on
 * error. timeout_ms < 0 waits indefinitely.
 */
int32_t kestrel_run_next_event(
    KestrelRun *run,
    int32_t timeout_ms,
    char **event_json,
    char **error);

/*
 * Returns 1 with output_json, 0 on timeout, or -1 on run/FFI error.
 * timeout_ms < 0 waits indefinitely.
 */
int32_t kestrel_run_wait(
    KestrelRun *run,
    int32_t timeout_ms,
    char **output_json,
    char **error);

int32_t kestrel_run_is_done(KestrelRun *run);
int32_t kestrel_run_cancel(KestrelRun *run);

/* Cancels and frees the run. May block until in-flight callbacks return. */
void kestrel_run_free(KestrelRun *run);

/* Cancels the active run, if any. Safe to call from another thread. */
int32_t kestrel_agent_cancel(KestrelAgent *agent);

/* Returns a kestrel::Session JSON snapshot. */
char *kestrel_agent_session_json(KestrelAgent *agent, char **error);

/* Returns the current todo list JSON, or NULL when todos are disabled. */
char *kestrel_agent_todos_json(KestrelAgent *agent, char **error);

int32_t kestrel_agent_reset(KestrelAgent *agent, char **error);

/* Deletes the stored session and clears local history. */
int32_t kestrel_agent_delete_session(KestrelAgent *agent, char **error);

#ifdef __cplusplus
}
#endif

#endif
