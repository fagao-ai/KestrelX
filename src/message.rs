use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Who produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Developer or application instructions.
    System,
    /// End-user input.
    User,
    /// Model output.
    Assistant,
    /// A tool result fed back to the model.
    Tool,
}

/// A piece of message content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain UTF-8 text.
    Text {
        /// Text payload.
        text: String,
    },
    /// A remote image URL.
    Image {
        /// Public or signed image URL.
        url: String,
        /// Optional image MIME type.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
}

impl ContentPart {
    /// Creates a text content part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Creates an image URL content part.
    pub fn image(url: impl Into<String>) -> Self {
        Self::Image {
            url: url.into(),
            mime_type: None,
        }
    }
}

/// A tool invocation requested by a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call identifier.
    pub id: String,
    /// Registered tool name.
    pub name: String,
    /// Decoded JSON arguments. Non-object values are rejected before execution.
    pub arguments: Value,
}

/// One conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Message author.
    pub role: Role,
    /// Message content. The default is empty, which is valid for an assistant
    /// message that only contains tool calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ContentPart>,
    /// Tool calls emitted by an assistant message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// The call this message answers, for [`Role::Tool`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Creates a system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self::simple(Role::System, text)
    }

    /// Creates a user message.
    pub fn user(text: impl Into<String>) -> Self {
        Self::simple(Role::User, text)
    }

    /// Creates an assistant message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::simple(Role::Assistant, text)
    }

    /// Creates a user message from content parts.
    pub fn user_with_content(content: Vec<ContentPart>) -> Self {
        Self {
            role: Role::User,
            content,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Creates an assistant message containing tool calls.
    pub fn assistant_with_tool_calls(text: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        let text = text.into();
        Self {
            role: Role::Assistant,
            content: if text.is_empty() {
                Vec::new()
            } else {
                vec![ContentPart::text(text)]
            },
            tool_calls,
            tool_call_id: None,
            name: None,
        }
    }

    /// Creates a tool result message.
    pub fn tool_result(call_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentPart::text(text)],
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
            name: None,
        }
    }

    /// Sets the optional tool name on this message.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Concatenates all text content in this message.
    pub fn text(&self) -> String {
        let mut result = String::new();
        for part in &self.content {
            if let ContentPart::Text { text } = part {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(text);
            }
        }
        result
    }

    /// Returns whether this message has no text and no tool calls.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty() && self.tool_calls.is_empty()
    }

    fn simple(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::text(text)],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        })
    }
}

impl From<&str> for Message {
    fn from(value: &str) -> Self {
        Self::user(value)
    }
}

impl From<String> for Message {
    fn from(value: String) -> Self {
        Self::user(value)
    }
}

impl From<Vec<ContentPart>> for Message {
    fn from(value: Vec<ContentPart>) -> Self {
        Self::user_with_content(value)
    }
}

/// Token accounting reported by a model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens.
    #[serde(default)]
    pub input_tokens: u32,
    /// Output tokens.
    #[serde(default)]
    pub output_tokens: u32,
}

impl Usage {
    /// Total tokens used.
    pub fn total(&self) -> u32 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.input_tokens = self.input_tokens.saturating_add(rhs.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(rhs.output_tokens);
    }
}

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural completion.
    Stop,
    /// Tool calls were requested.
    ToolCalls,
    /// Output token limit was reached.
    Length,
    /// A content filter stopped generation.
    ContentFilter,
    /// Provider-specific reason.
    Other(String),
}
