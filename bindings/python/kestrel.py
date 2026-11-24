"""Small ctypes binding for the Kestrel C ABI.

Build the shared library first:

    cargo build -p kestrel-ffi

Then set KESTREL_FFI_PATH if auto-discovery cannot find the library.
"""

from __future__ import annotations

import ctypes
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union


EventCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p)
ToolCallback = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
)
ModelCallback = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_char_p,
)
DeltaEmitter = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p)
StreamingModelCallback = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_char_p,
    DeltaEmitter,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
)
SessionLoadCallback = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p),
)
SessionSaveCallback = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p),
)
SessionDeleteCallback = SessionSaveCallback
MemoryRecallCallback = SessionLoadCallback
MemoryRememberCallback = SessionSaveCallback


class KestrelError(RuntimeError):
    """Raised when the native Kestrel library reports an error."""


def _candidate_libraries() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "libkestrel_ffi.dylib",
        "libkestrel_ffi.so",
        "kestrel_ffi.dll",
    )
    return [
        root / "target" / profile / name
        for profile in ("debug", "release")
        for name in names
    ]


def load_library(path: Optional[str] = None) -> ctypes.CDLL:
    """Loads the shared library and configures exported signatures."""
    candidates = [Path(path)] if path else []
    env_path = os.environ.get("KESTREL_FFI_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(_candidate_libraries())

    library_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if library_path is None:
        raise KestrelError(
            "kestrel_ffi shared library not found; run "
            "`cargo build -p kestrel-ffi` or set KESTREL_FFI_PATH"
        )

    library = ctypes.CDLL(str(library_path))
    library.kestrel_version.argtypes = []
    library.kestrel_version.restype = ctypes.c_char_p
    library.kestrel_contract_version.argtypes = []
    library.kestrel_contract_version.restype = ctypes.c_uint32
    library.kestrel_capabilities_json.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_capabilities_json.restype = ctypes.c_void_p
    library.kestrel_validate_config.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_validate_config.restype = ctypes.c_int32
    library.kestrel_string_new.argtypes = [ctypes.c_char_p]
    library.kestrel_string_new.restype = ctypes.c_void_p
    library.kestrel_string_free.argtypes = [ctypes.c_void_p]
    library.kestrel_string_free.restype = None
    library.kestrel_agent_new.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_new.restype = ctypes.c_void_p
    library.kestrel_agent_new_with_model.argtypes = [
        ctypes.c_char_p,
        ModelCallback,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_new_with_model.restype = ctypes.c_void_p
    library.kestrel_agent_new_with_streaming_model.argtypes = [
        ctypes.c_char_p,
        StreamingModelCallback,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_new_with_streaming_model.restype = ctypes.c_void_p
    library.kestrel_agent_builder_new.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_builder_new.restype = ctypes.c_void_p
    library.kestrel_agent_builder_new_with_model.argtypes = [
        ctypes.c_char_p,
        ModelCallback,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_builder_new_with_model.restype = ctypes.c_void_p
    library.kestrel_agent_builder_new_with_streaming_model.argtypes = [
        ctypes.c_char_p,
        StreamingModelCallback,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_builder_new_with_streaming_model.restype = ctypes.c_void_p
    library.kestrel_agent_builder_set_session_store.argtypes = [
        ctypes.c_void_p,
        SessionLoadCallback,
        SessionSaveCallback,
        SessionDeleteCallback,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_builder_set_session_store.restype = ctypes.c_int32
    library.kestrel_agent_builder_set_memory.argtypes = [
        ctypes.c_void_p,
        MemoryRecallCallback,
        MemoryRememberCallback,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_builder_set_memory.restype = ctypes.c_int32
    library.kestrel_agent_builder_build.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_builder_build.restype = ctypes.c_void_p
    library.kestrel_agent_builder_free.argtypes = [ctypes.c_void_p]
    library.kestrel_agent_builder_free.restype = None
    library.kestrel_agent_free.argtypes = [ctypes.c_void_p]
    library.kestrel_agent_free.restype = None
    library.kestrel_agent_set_event_callback.argtypes = [
        ctypes.c_void_p,
        EventCallback,
        ctypes.c_void_p,
    ]
    library.kestrel_agent_set_event_callback.restype = ctypes.c_int32
    library.kestrel_agent_register_tool.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ToolCallback,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_register_tool.restype = ctypes.c_int32
    library.kestrel_agent_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_run.restype = ctypes.c_void_p
    library.kestrel_agent_run_message.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_run_message.restype = ctypes.c_void_p
    library.kestrel_agent_start_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_start_run.restype = ctypes.c_void_p
    library.kestrel_agent_start_run_message.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_start_run_message.restype = ctypes.c_void_p
    library.kestrel_run_next_event.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_run_next_event.restype = ctypes.c_int32
    library.kestrel_run_wait.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_run_wait.restype = ctypes.c_int32
    library.kestrel_run_is_done.argtypes = [ctypes.c_void_p]
    library.kestrel_run_is_done.restype = ctypes.c_int32
    library.kestrel_run_cancel.argtypes = [ctypes.c_void_p]
    library.kestrel_run_cancel.restype = ctypes.c_int32
    library.kestrel_run_free.argtypes = [ctypes.c_void_p]
    library.kestrel_run_free.restype = None
    library.kestrel_agent_cancel.argtypes = [ctypes.c_void_p]
    library.kestrel_agent_cancel.restype = ctypes.c_int32
    library.kestrel_agent_session_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_session_json.restype = ctypes.c_void_p
    library.kestrel_agent_todos_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_todos_json.restype = ctypes.c_void_p
    library.kestrel_agent_reset.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.kestrel_agent_reset.restype = ctypes.c_int32
    return library


def _take_contract_string(library: ctypes.CDLL, pointer: int) -> str:
    value = ctypes.string_at(pointer).decode()
    library.kestrel_string_free(pointer)
    return value


def _take_contract_error(library: ctypes.CDLL, error: ctypes.c_void_p) -> str:
    if not error:
        return "unknown contract error"
    return _take_contract_string(library, error)


def contract_version(library: Optional[ctypes.CDLL] = None) -> int:
    """Returns the shared configuration schema version."""
    lib = library or load_library()
    return int(lib.kestrel_contract_version())


def capabilities(library: Optional[ctypes.CDLL] = None) -> list[str]:
    """Returns capability names supported by the FFI build."""
    lib = library or load_library()
    error = ctypes.c_void_p()
    pointer = lib.kestrel_capabilities_json(ctypes.byref(error))
    if not pointer:
        raise KestrelError(_take_contract_error(lib, error))
    return json.loads(_take_contract_string(lib, pointer))


def validate_config(
    config: Union[dict[str, Any], str],
    has_custom_model: bool = False,
    library: Optional[ctypes.CDLL] = None,
) -> None:
    """Validates config against the shared schema and FFI capabilities."""
    lib = library or load_library()
    config_json = config if isinstance(config, str) else json.dumps(config)
    error = ctypes.c_void_p()
    status = lib.kestrel_validate_config(
        config_json.encode(),
        1 if has_custom_model else 0,
        ctypes.byref(error),
    )
    if status != 0:
        raise KestrelError(_take_contract_error(lib, error))


class Agent:
    """Owns one native Kestrel agent and keeps callback objects alive."""

    def __init__(
        self,
        config: Union[dict[str, Any], str],
        *,
        model: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        stream_model: Optional[
            Callable[[dict[str, Any], Callable[[dict[str, Any]], None]], dict[str, Any]]
        ] = None,
        session_store: Optional[Any] = None,
        memory: Optional[Any] = None,
        library: Optional[ctypes.CDLL] = None,
    ) -> None:
        self._lib = library or load_library()
        self._handle = ctypes.c_void_p()
        self._tool_callbacks: list[Any] = []
        self._service_callbacks: list[Any] = []
        self._runs: list[Any] = []
        self._event_callback: Optional[Any] = None
        self.events: list[dict[str, Any]] = []
        self._event_handler: Optional[Callable[[dict[str, Any]], None]] = None
        config_json = config if isinstance(config, str) else json.dumps(config)
        error = ctypes.c_void_p()
        if model is not None and stream_model is not None:
            raise ValueError("model and stream_model are mutually exclusive")

        if session_store is not None or memory is not None:
            handle = self._build_with_services(
                config_json,
                model,
                stream_model,
                session_store,
                memory,
                error,
            )
        elif stream_model is not None:
            self._model_callback = StreamingModelCallback(
                self._streaming_model_proxy(stream_model)
            )
            handle = self._lib.kestrel_agent_new_with_streaming_model(
                config_json.encode(),
                self._model_callback,
                None,
                ctypes.byref(error),
            )
        elif model is None:
            self._model_callback = None
            handle = self._lib.kestrel_agent_new(config_json.encode(), ctypes.byref(error))
        else:
            self._model_callback = ModelCallback(self._model_proxy(model))
            handle = self._lib.kestrel_agent_new_with_model(
                config_json.encode(),
                self._model_callback,
                None,
                ctypes.byref(error),
            )

        if not handle:
            raise KestrelError(self._take_error(error))
        self._handle = ctypes.c_void_p(handle)

    @property
    def version(self) -> str:
        """Returns the native library version."""
        return self._lib.kestrel_version().decode()

    def set_event_handler(
        self,
        handler: Optional[Callable[[dict[str, Any]], None]],
    ) -> None:
        """Installs a Python event handler. Callbacks run on a Rust thread."""
        self._event_handler = handler
        if handler is None:
            self._event_callback = None
            self._lib.kestrel_agent_set_event_callback(self._handle, None, None)
            return

        def event_proxy(_user_data: Any, event_ptr: bytes) -> None:
            event = json.loads(event_ptr.decode())
            self.events.append(event)
            try:
                handler(event)
            except Exception:
                # Exceptions must not unwind through the C ABI.
                pass

        self._event_callback = EventCallback(event_proxy)
        status = self._lib.kestrel_agent_set_event_callback(
            self._handle,
            self._event_callback,
            None,
        )
        if status != 0:
            raise KestrelError("failed to install event callback")

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[
            [dict[str, Any], str],
            Union[str, dict[str, Any]],
        ],
    ) -> None:
        """Registers a synchronous Python tool callback."""
        def tool_proxy(
            _user_data: Any,
            call_id_ptr: bytes,
            arguments_ptr: bytes,
        ) -> int:
            try:
                arguments = json.loads(arguments_ptr.decode())
                result = handler(arguments, call_id_ptr.decode())
                text = result if isinstance(result, str) else json.dumps(result)
            except Exception as error:
                text = json.dumps(
                    {"content": f"Python tool failed: {error}", "is_error": True}
                )
            return int(self._lib.kestrel_string_new(text.encode()) or 0)

        callback = ToolCallback(tool_proxy)
        error = ctypes.c_void_p()
        status = self._lib.kestrel_agent_register_tool(
            self._handle,
            name.encode(),
            description.encode(),
            json.dumps(input_schema).encode(),
            callback,
            None,
            ctypes.byref(error),
        )
        if status != 0:
            raise KestrelError(self._take_error(error))
        self._tool_callbacks.append(callback)

    def run(self, prompt: str) -> dict[str, Any]:
        """Runs one prompt synchronously and returns AgentOutput JSON."""
        return self._run_call(
            self._lib.kestrel_agent_run,
            prompt.encode(),
        )

    def run_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Runs one user Message JSON object synchronously."""
        return self._run_call(
            self._lib.kestrel_agent_run_message,
            json.dumps(message).encode(),
        )

    def start_run(self, prompt: str) -> "Run":
        """Starts a nonblocking run and returns a pollable Run handle."""
        return self._start_run_call(
            self._lib.kestrel_agent_start_run,
            prompt.encode(),
        )

    def start_run_message(self, message: dict[str, Any]) -> "Run":
        """Starts a nonblocking run from a Message JSON object."""
        return self._start_run_call(
            self._lib.kestrel_agent_start_run_message,
            json.dumps(message).encode(),
        )

    def cancel(self) -> None:
        """Requests cancellation of the active run."""
        if self._handle:
            self._lib.kestrel_agent_cancel(self._handle)

    def session(self) -> dict[str, Any]:
        """Returns a session snapshot."""
        error = ctypes.c_void_p()
        pointer = self._lib.kestrel_agent_session_json(
            self._handle,
            ctypes.byref(error),
        )
        if not pointer:
            raise KestrelError(self._take_error(error))
        return json.loads(self._take_string(pointer))

    def todos(self) -> Optional[list[dict[str, Any]]]:
        """Returns todos, or None when todos are disabled for this agent."""
        error = ctypes.c_void_p()
        pointer = self._lib.kestrel_agent_todos_json(
            self._handle,
            ctypes.byref(error),
        )
        if not pointer:
            if not error:
                return None
            raise KestrelError(self._take_error(error))
        return json.loads(self._take_string(pointer))

    def reset(self) -> None:
        """Clears conversation history while keeping the session ID."""
        error = ctypes.c_void_p()
        status = self._lib.kestrel_agent_reset(
            self._handle,
            ctypes.byref(error),
        )
        if status != 0:
            raise KestrelError(self._take_error(error))

    def close(self) -> None:
        """Frees the native agent. Do not call while a run is active."""
        for run in list(self._runs):
            try:
                run.close()
            except Exception:
                pass
        self._runs.clear()
        if self._handle:
            self._lib.kestrel_agent_free(self._handle)
            self._handle = ctypes.c_void_p()
            self._tool_callbacks.clear()
            self._service_callbacks.clear()
            self._event_callback = None
            self._model_callback = None

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()

    def _run_call(self, function: Any, payload: bytes) -> dict[str, Any]:
        error = ctypes.c_void_p()
        pointer = function(self._handle, payload, ctypes.byref(error))
        if not pointer:
            raise KestrelError(self._take_error(error))
        return json.loads(self._take_string(pointer))

    def _start_run_call(self, function: Any, payload: bytes) -> "Run":
        error = ctypes.c_void_p()
        pointer = function(self._handle, payload, ctypes.byref(error))
        if not pointer:
            raise KestrelError(self._take_error(error))
        run = Run(self, pointer)
        self._runs.append(run)
        return run

    def _model_proxy(
        self,
        model: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> Callable[[Any, bytes], int]:
        def model_proxy(_user_data: Any, request_ptr: bytes) -> int:
            request = json.loads(request_ptr.decode())
            response = model(request)
            return self._allocate_json(response)

        return model_proxy

    def _build_with_services(
        self,
        config_json: str,
        model: Optional[Callable[[dict[str, Any]], dict[str, Any]]],
        stream_model: Optional[
            Callable[[dict[str, Any], Callable[[dict[str, Any]], None]], dict[str, Any]]
        ],
        session_store: Optional[Any],
        memory: Optional[Any],
        error: ctypes.c_void_p,
    ) -> int:
        if stream_model is not None:
            self._model_callback = StreamingModelCallback(
                self._streaming_model_proxy(stream_model)
            )
            builder = self._lib.kestrel_agent_builder_new_with_streaming_model(
                config_json.encode(),
                self._model_callback,
                None,
                ctypes.byref(error),
            )
        elif model is None:
            self._model_callback = None
            builder = self._lib.kestrel_agent_builder_new(
                config_json.encode(),
                ctypes.byref(error),
            )
        else:
            self._model_callback = ModelCallback(self._model_proxy(model))
            builder = self._lib.kestrel_agent_builder_new_with_model(
                config_json.encode(),
                self._model_callback,
                None,
                ctypes.byref(error),
            )
        if not builder:
            raise KestrelError(self._take_error(error))

        try:
            if session_store is not None:
                self._install_session_store(builder, session_store, error)
            if memory is not None:
                self._install_memory(builder, memory, error)
            handle = self._lib.kestrel_agent_builder_build(
                builder,
                ctypes.byref(error),
            )
        finally:
            self._lib.kestrel_agent_builder_free(builder)

        if not handle:
            raise KestrelError(self._take_error(error))
        return handle

    def _streaming_model_proxy(
        self,
        model: Callable[[dict[str, Any], Callable[[dict[str, Any]], None]], dict[str, Any]],
    ) -> Callable[[Any, bytes, Any, Any, Any], int]:
        def model_proxy(
            _user_data: Any,
            request_ptr: bytes,
            emit: Any,
            emit_ctx: Any,
            error_ptr: Any,
        ) -> int:
            try:
                request = json.loads(request_ptr.decode())

                def emit_delta(delta: dict[str, Any]) -> None:
                    emit(emit_ctx, json.dumps(delta).encode())

                return self._allocate_json(model(request, emit_delta))
            except Exception as callback_error:
                self._set_callback_error(error_ptr, callback_error)
                return 0

        return model_proxy

    def _install_session_store(
        self,
        builder: int,
        store: Any,
        error: ctypes.c_void_p,
    ) -> None:
        load_method = getattr(store, "load")
        save_method = getattr(store, "save")
        delete_method = getattr(store, "delete", None)

        def load_proxy(_user_data: Any, session_id: bytes, error_ptr: Any) -> int:
            try:
                session = load_method(session_id.decode())
                return 0 if session is None else self._allocate_json(session)
            except Exception as callback_error:
                self._set_callback_error(error_ptr, callback_error)
                return 0

        def save_proxy(_user_data: Any, session_json: bytes, error_ptr: Any) -> int:
            try:
                save_method(json.loads(session_json.decode()))
                return 0
            except Exception as callback_error:
                self._set_callback_error(error_ptr, callback_error)
                return -1

        load_callback = SessionLoadCallback(load_proxy)
        save_callback = SessionSaveCallback(save_proxy)
        callbacks: list[Any] = [load_callback, save_callback]
        delete_callback = None
        if delete_method is not None:
            def delete_proxy(
                _user_data: Any,
                session_id: bytes,
                error_ptr: Any,
            ) -> int:
                try:
                    delete_method(session_id.decode())
                    return 0
                except Exception as callback_error:
                    self._set_callback_error(error_ptr, callback_error)
                    return -1

            delete_callback = SessionDeleteCallback(delete_proxy)
            callbacks.append(delete_callback)

        status = self._lib.kestrel_agent_builder_set_session_store(
            builder,
            load_callback,
            save_callback,
            delete_callback,
            None,
            ctypes.byref(error),
        )
        if status != 0:
            raise KestrelError(self._take_error(error))
        self._service_callbacks.extend(callbacks)

    def _install_memory(
        self,
        builder: int,
        memory: Any,
        error: ctypes.c_void_p,
    ) -> None:
        recall_method = getattr(memory, "recall")
        remember_method = getattr(memory, "remember")

        def recall_proxy(_user_data: Any, query_json: bytes, error_ptr: Any) -> int:
            try:
                entries = recall_method(json.loads(query_json.decode()))
                return self._allocate_json(entries or [])
            except Exception as callback_error:
                self._set_callback_error(error_ptr, callback_error)
                return 0

        def remember_proxy(_user_data: Any, entry_json: bytes, error_ptr: Any) -> int:
            try:
                remember_method(json.loads(entry_json.decode()))
                return 0
            except Exception as callback_error:
                self._set_callback_error(error_ptr, callback_error)
                return -1

        recall_callback = MemoryRecallCallback(recall_proxy)
        remember_callback = MemoryRememberCallback(remember_proxy)
        status = self._lib.kestrel_agent_builder_set_memory(
            builder,
            recall_callback,
            remember_callback,
            None,
            ctypes.byref(error),
        )
        if status != 0:
            raise KestrelError(self._take_error(error))
        self._service_callbacks.extend([recall_callback, remember_callback])

    def _allocate_json(self, value: Any) -> int:
        encoded = json.dumps(value).encode()
        return int(self._lib.kestrel_string_new(encoded) or 0)

    def _set_callback_error(self, error_ptr: Any, error: Exception) -> None:
        if error_ptr:
            error_ptr[0] = self._lib.kestrel_string_new(str(error).encode())

    def _take_string(self, pointer: int) -> str:
        value = ctypes.string_at(pointer).decode()
        self._lib.kestrel_string_free(pointer)
        return value

    def _take_error(self, error: ctypes.c_void_p) -> str:
        if not error:
            return "unknown Kestrel FFI error"
        value = ctypes.string_at(error).decode()
        self._lib.kestrel_string_free(error)
        return value


class Run:
    """Pollable handle for one asynchronous native run."""

    def __init__(self, owner: Agent, handle: int) -> None:
        self._owner: Optional[Agent] = owner
        self._lib = owner._lib
        self._handle = ctypes.c_void_p(handle)
        self.ended = False

    @property
    def is_done(self) -> bool:
        """Returns whether the native run has produced its final outcome."""
        if not self._handle:
            return True
        return self._lib.kestrel_run_is_done(self._handle) == 1

    def next_event(self, timeout_ms: int = -1) -> Optional[dict[str, Any]]:
        """Returns the next event, None on timeout/end."""
        self._require_handle()
        event = ctypes.c_void_p()
        error = ctypes.c_void_p()
        status = self._lib.kestrel_run_next_event(
            self._handle,
            timeout_ms,
            ctypes.byref(event),
            ctypes.byref(error),
        )
        if status == 1:
            return json.loads(self._owner._take_string(event))
        if status == 0:
            return None
        if status == 2:
            self.ended = True
            return None
        raise KestrelError(self._owner._take_error(error))

    async def next_event_async(
        self,
        timeout_ms: int = -1,
    ) -> Optional[dict[str, Any]]:
        """Async wrapper around next_event."""
        return await asyncio.to_thread(self.next_event, timeout_ms)

    def wait(self, timeout_ms: int = -1) -> dict[str, Any]:
        """Waits for the final AgentOutput JSON."""
        self._require_handle()
        output = ctypes.c_void_p()
        error = ctypes.c_void_p()
        status = self._lib.kestrel_run_wait(
            self._handle,
            timeout_ms,
            ctypes.byref(output),
            ctypes.byref(error),
        )
        if status == 1:
            return json.loads(self._owner._take_string(output))
        if status == 0:
            raise TimeoutError("Kestrel run did not finish before timeout")
        raise KestrelError(self._owner._take_error(error))

    async def wait_async(self, timeout_ms: int = -1) -> dict[str, Any]:
        """Async wrapper around wait."""
        return await asyncio.to_thread(self.wait, timeout_ms)

    def cancel(self) -> None:
        """Requests cancellation without freeing the run handle."""
        if self._handle:
            self._lib.kestrel_run_cancel(self._handle)

    def close(self) -> None:
        """Cancels and frees the run, waiting for in-flight callbacks."""
        if self._handle:
            self._lib.kestrel_run_free(self._handle)
            self._handle = ctypes.c_void_p()
            self.ended = True
            self._owner = None

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()

    def _require_handle(self) -> None:
        if not self._handle or self._owner is None:
            raise KestrelError("Kestrel run is closed")
