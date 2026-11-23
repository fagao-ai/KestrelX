use crate::{ContentPart, Message, ToolSpec};

/// Estimates token usage without binding the core to a tokenizer dependency.
pub trait TokenCounter: Send + Sync {
    /// Estimates the number of tokens in plain text.
    fn count_text(&self, text: &str) -> usize;

    /// Estimates the tokens used by one message.
    fn count_message(&self, message: &Message) -> usize {
        let mut tokens = 4;
        tokens += self.count_text(&message.role.to_string());

        for part in &message.content {
            match part {
                ContentPart::Text { text } => tokens += self.count_text(text),
                ContentPart::Image { .. } => tokens += 850,
            }
        }
        for call in &message.tool_calls {
            tokens += 4;
            tokens += self.count_text(&call.name);
            tokens += self.count_text(&call.arguments.to_string());
        }
        if let Some(id) = &message.tool_call_id {
            tokens += self.count_text(id);
        }
        if let Some(name) = &message.name {
            tokens += self.count_text(name);
        }
        tokens
    }

    /// Estimates the tokens used by a message slice.
    fn count_messages(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|message| self.count_message(message))
            .sum()
    }

    /// Estimates prompt overhead for tool definitions.
    fn count_tools(&self, tools: &[ToolSpec]) -> usize {
        tools
            .iter()
            .map(|tool| {
                12 + self.count_text(&tool.name)
                    + self.count_text(&tool.description)
                    + self.count_text(&tool.input_schema.to_string())
            })
            .sum()
    }
}

/// A dependency-free approximation for mixed English and CJK text.
///
/// ASCII text is estimated at four characters per token. Every non-ASCII
/// character is counted as one token. This intentionally overestimates most
/// Western text and is a better fit for Chinese than a plain `chars / 4`
/// heuristic.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicTokenCounter;

impl TokenCounter for HeuristicTokenCounter {
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        let mut ascii = 0usize;
        let mut non_ascii = 0usize;
        for character in text.chars() {
            if character.is_ascii() {
                ascii += 1;
            } else {
                non_ascii += 1;
            }
        }
        ascii.div_ceil(4) + non_ascii
    }
}

impl TokenCounter for std::sync::Arc<dyn TokenCounter> {
    fn count_text(&self, text: &str) -> usize {
        self.as_ref().count_text(text)
    }

    fn count_message(&self, message: &Message) -> usize {
        self.as_ref().count_message(message)
    }

    fn count_messages(&self, messages: &[Message]) -> usize {
        self.as_ref().count_messages(messages)
    }

    fn count_tools(&self, tools: &[ToolSpec]) -> usize {
        self.as_ref().count_tools(tools)
    }
}
