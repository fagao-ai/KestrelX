use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    Compression, CompressionRequest, ContextCompressor, Error, Message, Result, TokenCounter,
};

use super::compress::{omission_marker, split_to_budget};

/// Keeps the newest atomic message groups that fit in the budget.
///
/// This is deterministic, dependency-free, and does not call a model. Older
/// history is replaced with a short omission marker.
#[derive(Debug, Clone)]
pub struct SlidingWindowCompressor {
    max_tokens: usize,
}

impl SlidingWindowCompressor {
    /// Creates a compressor with a fallback history budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens: max_tokens.max(1),
        }
    }
}

#[async_trait]
impl ContextCompressor for SlidingWindowCompressor {
    fn max_input_tokens(&self) -> Option<usize> {
        Some(self.max_tokens)
    }

    async fn compress(&self, request: CompressionRequest) -> Result<Compression> {
        compress_with_counter(request)
    }
}

fn compress_with_counter(request: CompressionRequest) -> Result<Compression> {
    let CompressionRequest {
        messages,
        budget_tokens,
        token_counter,
    } = request;
    let before_tokens = token_counter.count_messages(&messages);
    if before_tokens <= budget_tokens || messages.is_empty() {
        return Ok(Compression {
            messages,
            removed_messages: 0,
            summary: None,
            summarized: false,
            before_tokens,
            after_tokens: before_tokens,
        });
    }

    let marker_reserve = 24.min(budget_tokens / 4);
    let split = split_to_budget(
        &messages,
        budget_tokens.saturating_sub(marker_reserve),
        token_counter.as_ref(),
    );
    if split.removed_messages == 0 {
        return Err(Error::Context(format!(
            "the newest conversation group needs about {} tokens, but only {} are available",
            split.kept_tokens, budget_tokens
        )));
    }

    let marker = omission_marker(split.removed_messages);
    let mut compacted = Vec::with_capacity(split.kept.len() + 1);
    compacted.push(Message::system(marker.clone()));
    compacted.extend(split.kept);
    let after_tokens = token_counter.count_messages(&compacted);

    Ok(Compression {
        messages: compacted,
        removed_messages: split.removed_messages,
        summary: Some(marker),
        summarized: false,
        before_tokens,
        after_tokens,
    })
}

impl SlidingWindowCompressor {
    pub(crate) fn compress_with(
        &self,
        messages: Vec<Message>,
        budget_tokens: usize,
        token_counter: Arc<dyn TokenCounter>,
    ) -> Result<Compression> {
        compress_with_counter(CompressionRequest {
            messages,
            budget_tokens,
            token_counter,
        })
    }
}
