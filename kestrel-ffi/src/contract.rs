use std::ffi::{c_char, c_int};

use kestrel_contract::{AgentConfig, SCHEMA_VERSION};

use crate::{
    config, ffi_catch,
    strings::{clear_error, into_c_string, required_string, set_error},
};

/// Returns the shared configuration schema version.
#[unsafe(no_mangle)]
pub extern "C" fn kestrel_contract_version() -> u32 {
    SCHEMA_VERSION
}

/// Returns FFI capabilities as a JSON array of stable capability names.
///
/// # Safety
///
/// `error` must be NULL or point to writable storage for one `char *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_capabilities_json(error: *mut *mut c_char) -> *mut c_char {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        serde_json::to_string(&config::capabilities().names())
            .map_err(|error| format!("failed to encode capabilities: {error}"))
    });
    match result {
        Ok(json) => into_c_string(json),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            std::ptr::null_mut()
        }
    }
}

/// Validates config JSON against the shared contract and FFI capabilities.
///
/// Returns 0 on success and -1 on error. `has_custom_model` must be non-zero
/// when the caller will supply a model callback.
///
/// # Safety
///
/// Pointers follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_validate_config(
    config_json: *const c_char,
    has_custom_model: c_int,
    error: *mut *mut c_char,
) -> c_int {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: config_json is required by the C contract.
        let config_json = unsafe { required_string(config_json, "config_json") }?;
        AgentConfig::parse_and_validate_json(
            &config_json,
            has_custom_model != 0,
            &config::capabilities(),
        )
        .map_err(|error| error.to_string())?;
        Ok(())
    });
    match result {
        Ok(()) => 0,
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            -1
        }
    }
}
