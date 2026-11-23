#ifndef KESTREL_H
#define KESTREL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KestrelAgent KestrelAgent;

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

const char *kestrel_version(void);

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

/* Cancels the active run, if any. Safe to call from another thread. */
int32_t kestrel_agent_cancel(KestrelAgent *agent);

/* Returns a kestrel::Session JSON snapshot. */
char *kestrel_agent_session_json(KestrelAgent *agent, char **error);

/* Returns the current todo list JSON, or NULL when todos are disabled. */
char *kestrel_agent_todos_json(KestrelAgent *agent, char **error);

int32_t kestrel_agent_reset(KestrelAgent *agent, char **error);

#ifdef __cplusplus
}
#endif

#endif
