use std::{collections::BTreeMap, sync::Arc};

#[cfg(any(feature = "anthropic", feature = "gemini", feature = "openai"))]
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{Error, Model, Result};

/// Built-in model provider protocol.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    /// OpenAI Chat Completions and compatible servers.
    #[default]
    #[serde(rename = "openai_chat", alias = "openai")]
    OpenAiChat,
    /// OpenAI Responses API.
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    /// Anthropic Messages API.
    #[serde(rename = "anthropic")]
    Anthropic,
    /// Google Gemini Generative Language API.
    #[serde(rename = "gemini")]
    Gemini,
}

impl ProviderKind {
    /// Stable wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiChat => "openai_chat",
            Self::OpenAiResponses => "openai_responses",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
        }
    }
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Serialized configuration for one built-in HTTP provider.
///
/// Explicit values win over environment variables. API keys fall back to the
/// provider's conventional environment variables, matching the individual
/// provider builders.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderConfig {
    /// Wire protocol.
    pub kind: ProviderKind,
    /// Provider model identifier.
    pub model: String,
    /// API key. Empty values are treated as absent.
    pub api_key: Option<String>,
    /// API base URL. Provider defaults are used when omitted.
    pub base_url: Option<String>,
    /// Protocol version header, currently used by Anthropic.
    pub api_version: Option<String>,
    /// Context window used by automatic compression.
    pub context_window: Option<usize>,
    /// HTTP request timeout.
    pub timeout_seconds: Option<u64>,
    /// Request streaming usage, currently used by Chat Completions.
    pub stream_usage: Option<bool>,
    /// Additional HTTP headers.
    pub headers: BTreeMap<String, String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            kind: ProviderKind::OpenAiChat,
            model: String::new(),
            api_key: None,
            base_url: None,
            api_version: None,
            context_window: None,
            timeout_seconds: None,
            stream_usage: None,
            headers: BTreeMap::new(),
        }
    }
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("base_url", &self.base_url)
            .field("api_version", &self.api_version)
            .field("context_window", &self.context_window)
            .field("timeout_seconds", &self.timeout_seconds)
            .field("stream_usage", &self.stream_usage)
            .field("headers", &self.headers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ProviderConfig {
    /// Creates a provider configuration.
    pub fn new(kind: ProviderKind, model: impl Into<String>) -> Self {
        Self {
            kind,
            model: model.into(),
            ..Default::default()
        }
    }

    /// Validates common provider settings.
    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            return Err(Error::Configuration(
                "model name cannot be empty".to_owned(),
            ));
        }
        if self
            .base_url
            .as_ref()
            .is_some_and(|base_url| base_url.trim().is_empty())
        {
            return Err(Error::Configuration("base URL cannot be empty".to_owned()));
        }
        if self
            .api_version
            .as_ref()
            .is_some_and(|version| version.trim().is_empty())
        {
            return Err(Error::Configuration(
                "API version cannot be empty".to_owned(),
            ));
        }
        if self.context_window == Some(0) {
            return Err(Error::Configuration(
                "context window must be greater than zero".to_owned(),
            ));
        }
        if self.timeout_seconds == Some(0) {
            return Err(Error::Configuration(
                "timeout must be greater than zero".to_owned(),
            ));
        }
        Ok(())
    }

    /// Builds the configured provider.
    pub fn build(&self) -> Result<Arc<dyn Model>> {
        self.validate()?;
        match self.kind {
            ProviderKind::OpenAiChat => self.build_openai_chat(),
            ProviderKind::OpenAiResponses => self.build_openai_responses(),
            ProviderKind::Anthropic => self.build_anthropic(),
            ProviderKind::Gemini => self.build_gemini(),
        }
    }

    #[cfg(feature = "openai")]
    fn build_openai_chat(&self) -> Result<Arc<dyn Model>> {
        use crate::OpenAiCompatible;

        let mut builder = OpenAiCompatible::builder(&self.model);
        if let Some(api_key) = value_or_env(self.api_key.as_deref(), &["OPENAI_API_KEY"]) {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = value_or_env(self.base_url.as_deref(), &["OPENAI_BASE_URL"]) {
            builder = builder.base_url(base_url);
        }
        if let Some(tokens) = self.context_window {
            builder = builder.context_window(tokens);
        }
        if let Some(seconds) = self.timeout_seconds {
            builder = builder.timeout(Duration::from_secs(seconds));
        }
        if let Some(stream_usage) = self.stream_usage {
            builder = builder.stream_usage(stream_usage);
        }
        for (name, value) in &self.headers {
            builder = builder.header(name, value);
        }
        Ok(Arc::new(builder.build()?))
    }

    #[cfg(not(feature = "openai"))]
    fn build_openai_chat(&self) -> Result<Arc<dyn Model>> {
        self.feature_disabled("openai_chat", "openai")
    }

    #[cfg(feature = "openai")]
    fn build_openai_responses(&self) -> Result<Arc<dyn Model>> {
        use crate::OpenAiResponses;

        let mut builder = OpenAiResponses::builder(&self.model);
        if let Some(api_key) = value_or_env(self.api_key.as_deref(), &["OPENAI_API_KEY"]) {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = value_or_env(self.base_url.as_deref(), &["OPENAI_BASE_URL"]) {
            builder = builder.base_url(base_url);
        }
        if let Some(tokens) = self.context_window {
            builder = builder.context_window(tokens);
        }
        if let Some(seconds) = self.timeout_seconds {
            builder = builder.timeout(Duration::from_secs(seconds));
        }
        for (name, value) in &self.headers {
            builder = builder.header(name, value);
        }
        Ok(Arc::new(builder.build()?))
    }

    #[cfg(not(feature = "openai"))]
    fn build_openai_responses(&self) -> Result<Arc<dyn Model>> {
        self.feature_disabled("openai_responses", "openai")
    }

    #[cfg(feature = "anthropic")]
    fn build_anthropic(&self) -> Result<Arc<dyn Model>> {
        use crate::Anthropic;

        let mut builder = Anthropic::builder(&self.model);
        if let Some(api_key) = value_or_env(self.api_key.as_deref(), &["ANTHROPIC_API_KEY"]) {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = value_or_env(self.base_url.as_deref(), &["ANTHROPIC_BASE_URL"]) {
            builder = builder.base_url(base_url);
        }
        if let Some(version) = value_or_env(self.api_version.as_deref(), &["ANTHROPIC_VERSION"]) {
            builder = builder.version(version);
        }
        if let Some(tokens) = self.context_window {
            builder = builder.context_window(tokens);
        }
        if let Some(seconds) = self.timeout_seconds {
            builder = builder.timeout(Duration::from_secs(seconds));
        }
        for (name, value) in &self.headers {
            builder = builder.header(name, value);
        }
        Ok(Arc::new(builder.build()?))
    }

    #[cfg(not(feature = "anthropic"))]
    fn build_anthropic(&self) -> Result<Arc<dyn Model>> {
        self.feature_disabled("anthropic", "anthropic")
    }

    #[cfg(feature = "gemini")]
    fn build_gemini(&self) -> Result<Arc<dyn Model>> {
        use crate::Gemini;

        let mut builder = Gemini::builder(&self.model);
        if let Some(api_key) = value_or_env(
            self.api_key.as_deref(),
            &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        ) {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = value_or_env(self.base_url.as_deref(), &["GEMINI_BASE_URL"]) {
            builder = builder.base_url(base_url);
        }
        if let Some(tokens) = self.context_window {
            builder = builder.context_window(tokens);
        }
        if let Some(seconds) = self.timeout_seconds {
            builder = builder.timeout(Duration::from_secs(seconds));
        }
        for (name, value) in &self.headers {
            builder = builder.header(name, value);
        }
        Ok(Arc::new(builder.build()?))
    }

    #[cfg(not(feature = "gemini"))]
    fn build_gemini(&self) -> Result<Arc<dyn Model>> {
        self.feature_disabled("gemini", "gemini")
    }

    #[cfg(any(
        not(feature = "openai"),
        not(feature = "anthropic"),
        not(feature = "gemini")
    ))]
    fn feature_disabled(&self, provider: &str, feature: &str) -> Result<Arc<dyn Model>> {
        Err(Error::Configuration(format!(
            "provider `{provider}` requires the `{feature}` feature"
        )))
    }
}

#[cfg(any(feature = "anthropic", feature = "gemini", feature = "openai"))]
fn value_or_env(value: Option<&str>, variables: &[&str]) -> Option<String> {
    value
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            variables.iter().find_map(|variable| {
                std::env::var(variable)
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_provider_kinds_and_defaults() {
        let config: ProviderConfig =
            serde_json::from_str(r#"{"kind":"anthropic","model":"claude-test"}"#).unwrap();
        assert_eq!(config.kind, ProviderKind::Anthropic);
        assert_eq!(config.timeout_seconds, None);

        let config: ProviderConfig =
            serde_json::from_str(r#"{"kind":"openai","model":"gpt-test"}"#).unwrap();
        assert_eq!(config.kind, ProviderKind::OpenAiChat);
    }

    #[test]
    fn validates_provider_limits() {
        let mut config = ProviderConfig::new(ProviderKind::Gemini, "gemini-test");
        assert!(config.validate().is_ok());

        config.context_window = Some(0);
        assert!(config.validate().is_err());
        config.context_window = None;
        config.timeout_seconds = Some(0);
        assert!(config.validate().is_err());
    }
}
