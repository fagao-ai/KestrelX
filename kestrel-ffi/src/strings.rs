use std::ffi::{CStr, CString, c_char};

/// Allocates a copy of `value` with Rust's allocator.
///
/// # Safety
///
/// `value` must be NULL or a valid NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_string_new(value: *const c_char) -> *mut c_char {
    if value.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: the caller guarantees a valid NUL-terminated C string.
    let value = unsafe { CStr::from_ptr(value) };
    match CString::new(value.to_bytes()) {
        Ok(string) => string.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Frees a string allocated by Kestrel or `kestrel_string_new`.
///
/// # Safety
///
/// `value` must be NULL or an unreleased pointer returned by Kestrel's string
/// allocator. Each allocation may be freed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_string_free(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    // SAFETY: ownership is transferred exactly once by the caller.
    drop(unsafe { CString::from_raw(value) });
}

pub(crate) fn into_c_string(value: String) -> *mut c_char {
    let sanitized = value.replace('\0', "\\0");
    CString::new(sanitized)
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

pub(crate) unsafe fn set_error(out_error: *mut *mut c_char, message: String) {
    if out_error.is_null() {
        return;
    }
    // SAFETY: caller provides a writable char** location.
    unsafe {
        *out_error = into_c_string(message);
    }
}

pub(crate) unsafe fn clear_error(out_error: *mut *mut c_char) {
    if !out_error.is_null() {
        // SAFETY: caller provides a writable char** location.
        unsafe {
            *out_error = std::ptr::null_mut();
        }
    }
}

pub(crate) unsafe fn required_string(value: *const c_char, name: &str) -> Result<String, String> {
    if value.is_null() {
        return Err(format!("{name} must not be NULL"));
    }
    // SAFETY: caller guarantees a NUL-terminated C string.
    let value = unsafe { CStr::from_ptr(value) };
    value
        .to_str()
        .map(str::to_owned)
        .map_err(|error| format!("{name} is not valid UTF-8: {error}"))
}
