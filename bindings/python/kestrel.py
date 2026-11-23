"""Small ctypes binding for the Kestrel C ABI.

Build the shared library first:

    cargo build -p kestrel-ffi

Then set KESTREL_FFI_PATH if auto-discovery cannot find the library.
"""

from __future__ import annotations

import ctypes
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


class Agent:
    """Owns one native Kestrel agent and keeps callback objects alive."""

    def __init__(
        self,
        config: Union[dict[str, Any], str],
        *,
        model: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        library: Optional[ctypes.CDLL] = None,
    ) -> None:
        self._lib = library or load_library()
        self._handle = ctypes.c_void_p()
        self._tool_callbacks: list[Any] = []
        self._event_callback: Optional[Any] = None
        self.events: list[dict[str, Any]] = []
        self._event_handler: Optional[Callable[[dict[str, Any]], None]] = None
        config_json = config if isinstance(config, str) else json.dumps(config)
        error = ctypes.c_void_p()

        if model is None:
            self._model_callback = None
            handle = self._lib.kestrel_agent_new(
                config_json.encode(),
                ctypes.byref(error),
            )
        else:
            def model_proxy(_user_data: Any, request_ptr: bytes) -> int:
                request = json.loads(request_ptr.decode())
                response = model(request)
                encoded = json.dumps(response).encode()
                return int(self._lib.kestrel_string_new(encoded) or 0)

            self._model_callback = ModelCallback(model_proxy)
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
        if self._handle:
            self._lib.kestrel_agent_free(self._handle)
            self._handle = ctypes.c_void_p()
            self._tool_callbacks.clear()
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
