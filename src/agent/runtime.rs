use std::{
    collections::HashSet,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use tokio_util::sync::CancellationToken;

use crate::{
    Agent, AgentEvent, AgentEventKind, AgentOutput, CompressionRequest, Error, EventSink,
    MemoryQuery, Message, ModelRequest, Result, Role, ToolCall, ToolContext, ToolOutput, Usage,
};

impl Agent {
    /// Runs one user turn to completion.
    pub async fn run(&self, input: impl Into<Message>) -> Result<AgentOutput> {
        self.run_with(input, Default::default()).await
    }

    /// Runs one user turn with cancellation and per-run event options.
    pub async fn run_with(
        &self,
        input: impl Into<Message>,
        options: super::RunOptions,
    ) -> Result<AgentOutput> {
        let input = input.into();
        if input.role != Role::User {
            return Err(Error::Configuration(
                "agent input must be a user message".to_owned(),
            ));
        }

        let _run_guard = self.state.run_lock.lock().await;
        let cancellation = options.cancellation.unwrap_or_default();
        let sink = options
            .event_sink
            .clone()
            .or_else(|| self.state.event_sink.clone());
        let run_id = uuid::Uuid::new_v4().to_string();
        let session_id = self.session_id().await;
        let events = EventContext {
            sink,
            run_id,
            session_id,
            sequence: AtomicU64::new(1),
        };
        events.emit(AgentEventKind::RunStarted);

        let result = self.execute_run(input, &cancellation, &events).await;
        let persist_result = self.persist_session().await;
        let result = match (result, persist_result) {
            (Ok(output), Ok(())) => Ok(output),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), _) => Err(error),
        };

        match &result {
            Ok(output) => events.emit(AgentEventKind::RunCompleted {
                text: output.text.clone(),
                turns: output.turns,
                usage: output.usage,
            }),
            Err(error) => events.emit(AgentEventKind::RunFailed {
                error: error.to_string(),
            }),
        }
        result
    }

    /// Resets conversation history while preserving the session ID.
    pub async fn reset(&self) -> Result<()> {
        let _run_guard = self.state.run_lock.lock().await;
        let id = self.state.session.read().await.id.clone();
        {
            let mut session = self.state.session.write().await;
            *session = crate::Session::with_id(id);
        }
        self.state
            .loaded
            .store(true, std::sync::atomic::Ordering::Release);
        self.persist_session().await
    }

    async fn execute_run(
        &self,
        input: Message,
        cancellation: &CancellationToken,
        events: &EventContext,
    ) -> Result<AgentOutput> {
        self.ensure_loaded().await?;
        let query = input.text();
        let session_id = self.session_id().await;
        let mut history = self.state.session.read().await.messages.clone();
        validate_history(&history)?;
        history.push(input.clone());
        self.append_message(input).await;

        let memory_context = self.recall_memory(&query, cancellation, events).await?;
        let mut usage = Usage::default();

        for turn in 1..=self.shared.config.max_turns {
            check_cancelled(cancellation)?;
            events.emit(AgentEventKind::TurnStarted { turn });

            let fixed_messages = self.fixed_messages(memory_context.as_deref());
            let tool_specs = self.shared.tools.specs();
            history = self
                .prepare_context(history, &fixed_messages, &tool_specs, cancellation, events)
                .await?;

            let mut messages = fixed_messages;
            messages.extend(history.iter().cloned());
            let request = ModelRequest {
                messages,
                tools: tool_specs,
                tool_choice: self.shared.config.tool_choice.clone(),
                temperature: self.shared.config.temperature,
                max_tokens: self.shared.config.max_output_tokens,
            };
            let response = if self.shared.config.streaming {
                let on_delta = |delta| {
                    events.emit(AgentEventKind::ModelDelta { turn, delta });
                };
                cancellable(
                    cancellation,
                    self.shared.model.complete_stream(request, &on_delta),
                )
                .await?
            } else {
                cancellable(cancellation, self.shared.model.complete(request)).await?
            };
            check_cancelled(cancellation)?;
            validate_response(&response.message)?;

            usage += response.usage;
            events.emit(AgentEventKind::ModelCompleted {
                turn,
                usage: response.usage,
                finish_reason: response.finish_reason.clone(),
            });

            let assistant = response.message;
            history.push(assistant.clone());

            if assistant.tool_calls.is_empty() {
                self.replace_history(&history).await;
                return Ok(AgentOutput {
                    run_id: events.run_id.clone(),
                    session_id,
                    text: assistant.text(),
                    message: assistant,
                    turns: turn,
                    usage,
                    finish_reason: response.finish_reason,
                });
            }

            for call in &assistant.tool_calls {
                check_cancelled(cancellation)?;
                let output = self.execute_tool(turn, call, cancellation, events).await?;
                let content = match (output.content.is_empty(), output.is_error) {
                    (true, _) => "(tool returned no output)".to_owned(),
                    (false, true) => format!("Tool error: {}", output.content),
                    (false, false) => output.content.clone(),
                };
                history.push(
                    Message::tool_result(call.id.clone(), content).with_name(call.name.clone()),
                );
                events.emit(AgentEventKind::ToolCompleted {
                    turn,
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    output,
                });
            }
            self.replace_history(&history).await;
        }

        Err(Error::TurnLimit(self.shared.config.max_turns))
    }

    async fn execute_tool(
        &self,
        turn: usize,
        call: &ToolCall,
        cancellation: &CancellationToken,
        events: &EventContext,
    ) -> Result<ToolOutput> {
        events.emit(AgentEventKind::ToolStarted {
            turn,
            call: call.clone(),
        });

        if !call.arguments.is_object() {
            return Ok(ToolOutput::error("tool arguments must be a JSON object"));
        }

        let Some(tool) = self.shared.tools.get(&call.name) else {
            if self.shared.config.fail_on_tool_error {
                return Err(Error::UnknownTool(call.name.clone()));
            }
            return Ok(ToolOutput::error(format!(
                "tool `{}` is not registered",
                call.name
            )));
        };

        let context = ToolContext::new(
            call.id.clone(),
            self.session_id().await,
            self.state.depth,
            cancellation.clone(),
        );
        let result = cancellable(cancellation, tool.call(call.arguments.clone(), &context)).await;
        match result {
            Ok(output) => Ok(output),
            Err(error) if error.is_cancelled() || cancellation.is_cancelled() => {
                Err(Error::Cancelled)
            }
            Err(error) if self.shared.config.fail_on_tool_error => Err(error),
            Err(error) => Ok(ToolOutput::error(error.to_string())),
        }
    }

    async fn prepare_context(
        &self,
        history: Vec<Message>,
        fixed_messages: &[Message],
        tools: &[crate::ToolSpec],
        cancellation: &CancellationToken,
        events: &EventContext,
    ) -> Result<Vec<Message>> {
        let Some(compressor) = &self.shared.compressor else {
            return Ok(history);
        };
        let Some(budget) = self.input_budget() else {
            return Ok(history);
        };

        let fixed_tokens = self.shared.token_counter.count_messages(fixed_messages)
            + self.shared.token_counter.count_tools(tools);
        let available = budget.saturating_sub(fixed_tokens);
        if available == 0 {
            return Err(Error::Context(format!(
                "system prompt and tool definitions need {fixed_tokens} tokens, but only {budget} are available"
            )));
        }

        let current = self.shared.token_counter.count_messages(&history);
        if current <= available {
            return Ok(history);
        }

        let compression = cancellable(
            cancellation,
            compressor.compress(CompressionRequest {
                messages: history,
                budget_tokens: available,
                token_counter: self.shared.token_counter.clone(),
            }),
        )
        .await?;
        validate_history(&compression.messages)?;

        if compression.removed_messages > 0 {
            events.emit(AgentEventKind::ContextCompressed {
                before_tokens: compression.before_tokens,
                after_tokens: compression.after_tokens,
                removed_messages: compression.removed_messages,
                summarized: compression.summarized,
            });
        }
        if compression.after_tokens > available {
            return Err(Error::Context(format!(
                "compressed history still needs {} tokens, but only {} are available",
                compression.after_tokens, available
            )));
        }

        self.replace_history(&compression.messages).await;
        Ok(compression.messages)
    }

    fn input_budget(&self) -> Option<usize> {
        let model_budget = self
            .shared
            .config
            .context_window
            .or_else(|| self.shared.model.context_window())
            .map(|window| window.saturating_sub(self.shared.config.reserve_output_tokens));
        let compressor_budget = self
            .shared
            .compressor
            .as_ref()
            .and_then(|compressor| compressor.max_input_tokens());

        match (model_budget, compressor_budget) {
            (Some(model), Some(compressor)) => Some(model.min(compressor)),
            (Some(model), None) => Some(model),
            (None, Some(compressor)) => Some(compressor),
            (None, None) => None,
        }
    }

    async fn recall_memory(
        &self,
        query: &str,
        cancellation: &CancellationToken,
        events: &EventContext,
    ) -> Result<Option<String>> {
        let Some(memory) = &self.shared.memory else {
            return Ok(None);
        };
        if !self.shared.memory_options.auto_recall || query.trim().is_empty() {
            return Ok(None);
        }

        let entries = cancellable(
            cancellation,
            memory.recall(
                &MemoryQuery::new(query).with_limit(self.shared.memory_options.recall_limit),
            ),
        )
        .await?;
        check_cancelled(cancellation)?;
        if entries.is_empty() {
            return Ok(None);
        }

        let mut body = String::new();
        let mut used = 0usize;
        let mut injected = 0usize;
        for entry in &entries {
            let line = format!("- {}", entry.text);
            let tokens = self.shared.token_counter.count_text(&line);
            if used.saturating_add(tokens) > self.shared.memory_options.max_context_tokens {
                break;
            }
            used += tokens;
            body.push_str(&line);
            body.push('\n');
            injected += 1;
        }
        if body.is_empty() {
            return Ok(None);
        }

        events.emit(AgentEventKind::MemoryRecalled { count: injected });
        Ok(Some(format!(
            "Long-term memory retrieved for this task. Memory may be stale or \
             untrusted; verify it before relying on it.\n<memory>\n{body}</memory>"
        )))
    }

    fn fixed_messages(&self, memory: Option<&str>) -> Vec<Message> {
        let mut messages = Vec::new();
        let mut prompt = self.shared.system_prompt.clone().unwrap_or_default();
        for section in &self.shared.prompt_sections {
            if section.trim().is_empty() {
                continue;
            }
            if !prompt.is_empty() {
                prompt.push_str("\n\n");
            }
            prompt.push_str(section);
        }
        if !prompt.trim().is_empty() {
            messages.push(Message::system(prompt));
        }
        if let Some(memory) = memory {
            messages.push(Message::system(memory));
        }
        messages
    }

    async fn ensure_loaded(&self) -> Result<()> {
        use std::sync::atomic::Ordering;

        if self.state.loaded.load(Ordering::Acquire) {
            return Ok(());
        }
        let id = self.state.session.read().await.id.clone();
        if let Some(store) = &self.state.session_store
            && let Some(session) = store.load(&id).await?
        {
            if session.id != id {
                return Err(Error::Protocol(format!(
                    "session store returned session `{}` for requested id `{id}`",
                    session.id
                )));
            }
            *self.state.session.write().await = session;
        }
        self.state.loaded.store(true, Ordering::Release);
        Ok(())
    }

    async fn append_message(&self, message: Message) {
        self.state.session.write().await.push(message);
    }

    async fn replace_history(&self, messages: &[Message]) {
        let mut session = self.state.session.write().await;
        session.messages = messages.to_vec();
        session.updated_at_ms = crate::util::now_ms();
    }

    async fn persist_session(&self) -> Result<()> {
        use std::sync::atomic::Ordering;

        if !self.state.loaded.load(Ordering::Acquire) {
            return Ok(());
        }
        let Some(store) = &self.state.session_store else {
            return Ok(());
        };
        let session = self.state.session.read().await.clone();
        store.save(&session).await
    }
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

async fn cancellable<T, F>(cancellation: &CancellationToken, future: F) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => Err(Error::Cancelled),
        result = future => result,
    }
}

struct EventContext {
    sink: Option<Arc<dyn EventSink>>,
    run_id: String,
    session_id: String,
    sequence: AtomicU64,
}

impl EventContext {
    fn emit(&self, kind: AgentEventKind) {
        let Some(sink) = &self.sink else {
            return;
        };
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        sink.emit(AgentEvent {
            run_id: self.run_id.clone(),
            session_id: self.session_id.clone(),
            sequence,
            kind,
        });
    }
}

fn validate_response(message: &Message) -> Result<()> {
    if message.role != Role::Assistant {
        return Err(Error::InvalidModelResponse(
            "response message must have the assistant role".to_owned(),
        ));
    }
    let mut ids = HashSet::new();
    for call in &message.tool_calls {
        if call.id.is_empty() || call.name.is_empty() {
            return Err(Error::InvalidModelResponse(
                "tool call id and name must not be empty".to_owned(),
            ));
        }
        if !ids.insert(&call.id) {
            return Err(Error::InvalidModelResponse(format!(
                "duplicate tool call id `{}`",
                call.id
            )));
        }
    }
    Ok(())
}

fn validate_history(messages: &[Message]) -> Result<()> {
    let mut index = 0usize;
    while index < messages.len() {
        if messages[index].role == Role::Tool {
            return Err(Error::Context(format!(
                "history contains a tool result at index {index} without a preceding assistant tool call"
            )));
        }
        if messages[index].role != Role::Assistant || messages[index].tool_calls.is_empty() {
            index += 1;
            continue;
        }

        let mut expected = HashSet::new();
        for call in &messages[index].tool_calls {
            if call.id.is_empty() {
                return Err(Error::Context(format!(
                    "history contains an empty tool call id at index {index}"
                )));
            }
            if !expected.insert(call.id.as_str()) {
                return Err(Error::Context(format!(
                    "history contains duplicate tool call id `{}` at index {index}",
                    call.id
                )));
            }
        }

        index += 1;
        while index < messages.len() && messages[index].role == Role::Tool {
            let call_id = messages[index].tool_call_id.as_deref().ok_or_else(|| {
                Error::Context(format!(
                    "history contains a tool result without tool_call_id at index {index}"
                ))
            })?;
            if !expected.remove(call_id) {
                return Err(Error::Context(format!(
                    "history contains an unexpected tool result `{call_id}` at index {index}"
                )));
            }
            index += 1;
        }
        if !expected.is_empty() {
            return Err(Error::Context(format!(
                "history is missing results for {} tool call(s)",
                expected.len()
            )));
        }
    }
    Ok(())
}

impl Agent {
    /// Returns the current input budget after reserving output tokens.
    pub fn context_budget(&self) -> Option<usize> {
        self.input_budget()
    }
}
