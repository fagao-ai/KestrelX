use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    Compression, CompressionRequest, ContextCompressor, Error, FinishReason, Message, Model,
    ModelRequest, Result, ToolChoice,
};

use super::{
    SlidingWindowCompressor,
    compress::{render_for_summary, split_to_budget, summary_message, truncate_text_to_tokens},
};

/// Compresses old history into a model-generated summary.
///
/// The newest messages stay verbatim. If summarization fails, the compressor
/// can fall back to [`SlidingWindowCompressor`]-style truncation.
pub struct SummarizingCompressor {
    model: Arc<dyn Model>,
    max_tokens: usize,
    summary_tokens: usize,
    fallback_to_sliding: bool,
}

impl std::fmt::Debug for SummarizingCompressor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SummarizingCompressor")
            .field("model", &self.model.name())
            .field("max_tokens", &self.max_tokens)
            .field("summary_tokens", &self.summary_tokens)
            .field("fallback_to_sliding", &self.fallback_to_sliding)
            .finish()
    }
}

impl SummarizingCompressor {
    /// Creates a summarizer using `model` for summaries.
    pub fn new(model: Arc<dyn Model>, max_tokens: usize) -> Self {
        Self {
            model,
            max_tokens: max_tokens.max(1),
            summary_tokens: 512,
            fallback_to_sliding: true,
        }
    }

    /// Sets the maximum size of the generated summary.
    pub fn with_summary_tokens(mut self, tokens: usize) -> Self {
        self.summary_tokens = tokens.max(32);
        self
    }

    /// Enables or disables deterministic truncation on summarizer failure.
    pub fn with_fallback(mut self, enabled: bool) -> Self {
        self.fallback_to_sliding = enabled;
        self
    }

    async fn summarize(&self, dropped: &[Message], budget: usize) -> Result<String> {
        let transcript = render_for_summary(dropped);
        let response = self
            .model
            .complete(ModelRequest {
                messages: vec![
                    Message::system(
                        "Compress conversation history for another assistant. Preserve user \
                         requirements, decisions, facts, identifiers, tool results, and open \
                         tasks. Do not invent details. Return only the summary.",
                    ),
                    Message::user(transcript),
                ],
                tools: Vec::new(),
                tool_choice: ToolChoice::None,
                temperature: Some(0.0),
                max_tokens: Some(budget.min(u32::MAX as usize) as u32),
            })
            .await
            .map_err(|error| Error::Model(format!("context summarization failed: {error}")))?;

        if !matches!(
            response.finish_reason,
            FinishReason::Stop | FinishReason::Length
        ) {
            return Err(Error::Model(
                "summarizer returned an unsupported finish reason".to_owned(),
            ));
        }

        let summary = response.message.text().trim().to_owned();
        if summary.is_empty() {
            Err(Error::Model(
                "summarizer returned an empty response".to_owned(),
            ))
        } else {
            Ok(summary)
        }
    }
}

#[async_trait]
impl ContextCompressor for SummarizingCompressor {
    fn max_input_tokens(&self) -> Option<usize> {
        Some(self.max_tokens)
    }

    async fn compress(&self, request: CompressionRequest) -> Result<Compression> {
        let before_tokens = request.token_counter.count_messages(&request.messages);
        if before_tokens <= request.budget_tokens || request.messages.is_empty() {
            return Ok(Compression {
                messages: request.messages,
                removed_messages: 0,
                summary: None,
                summarized: false,
                before_tokens,
                after_tokens: before_tokens,
            });
        }

        let summary_reserve = self.summary_tokens.min(request.budget_tokens / 3);
        let split = split_to_budget(
            &request.messages,
            request.budget_tokens.saturating_sub(summary_reserve),
            request.token_counter.as_ref(),
        );

        if split.removed_messages == 0 {
            if self.fallback_to_sliding {
                return SlidingWindowCompressor::new(request.budget_tokens).compress_with(
                    request.messages,
                    request.budget_tokens,
                    request.token_counter,
                );
            }
            return Err(Error::Context(format!(
                "the newest conversation group needs about {} tokens, but only {} are available",
                split.kept_tokens, request.budget_tokens
            )));
        }

        let dropped = &request.messages[..split.removed_messages];
        let generated = self.summarize(dropped, self.summary_tokens).await;
        let (summary, summarized) = match generated {
            Ok(summary) => (summary, true),
            Err(_) if self.fallback_to_sliding => {
                let fallback = SlidingWindowCompressor::new(request.budget_tokens).compress_with(
                    request.messages,
                    request.budget_tokens,
                    request.token_counter.clone(),
                )?;
                return Ok(fallback);
            }
            Err(error) => return Err(error),
        };

        let available = request
            .budget_tokens
            .saturating_sub(split.kept_tokens)
            .saturating_sub(12);
        let summary = truncate_text_to_tokens(
            &summary,
            available.min(self.summary_tokens),
            request.token_counter.as_ref(),
        );
        if summary.is_empty() {
            return Err(Error::Context(
                "no budget remains for a conversation summary".to_owned(),
            ));
        }

        let mut compacted = Vec::with_capacity(split.kept.len() + 1);
        compacted.push(summary_message(&summary));
        compacted.extend(split.kept);
        let after_tokens = request.token_counter.count_messages(&compacted);
        if after_tokens > request.budget_tokens {
            if self.fallback_to_sliding {
                return SlidingWindowCompressor::new(request.budget_tokens).compress_with(
                    request.messages,
                    request.budget_tokens,
                    request.token_counter,
                );
            }
            return Err(Error::Context(format!(
                "summary plus recent history needs {after_tokens} tokens, but only {} are available",
                request.budget_tokens
            )));
        }

        Ok(Compression {
            messages: compacted,
            removed_messages: split.removed_messages,
            summary: Some(summary),
            summarized,
            before_tokens,
            after_tokens,
        })
    }
}
