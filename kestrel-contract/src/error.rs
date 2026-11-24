/// Configuration contract errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ContractError {
    /// The caller requested an unknown or unsupported schema version.
    #[error("unsupported schema_version {found}; expected {expected}")]
    UnsupportedSchema {
        /// Found version.
        found: u32,
        /// Supported version.
        expected: u32,
    },

    /// The configuration is structurally invalid.
    #[error("invalid Kestrel config: {0}")]
    Invalid(String),

    /// The binding does not implement one or more required capabilities.
    #[error("binding does not support required capabilities: {0}")]
    UnsupportedCapabilities(String),

    /// JSON parsing failed.
    #[error("invalid Kestrel config JSON: {0}")]
    Json(#[from] serde_json::Error),
}
