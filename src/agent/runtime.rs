use std::{
    collections::HashSet,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::{
    Agent, AgentEvent, AgentEventKind, AgentOutput, CompressionRequest, Error, EventSink,
    MemoryQuery, Message, ModelRequest, Result, Role, RunCheckpoint, RunPhase,
    SESSION_SCHEMA_VERSION, ToolCall, ToolContext, ToolOutput, Usage,
};

impl Agent {
    /// Runs one user turn to completion.
    ///
    /// Returns [`Error::ActiveRun`] when the session contains an interrupted
    /// run. Call [`Agent::resume`] to continue it or reset the session first.
    pub async fn run(&self, input: impl Into<Message>) -> Result<AgentOutput> {
        self.run_with(input, Default::default()).await
    }

    /// Runs one user turn with cancellation and per-run event options.
    ///
    /// Returns [`Error::ActiveRun`] when the session contains an interrupted
    /// run. Call [`Agent::resume_with`] to continue it or reset the session first.
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

        let cancellation = options.cancellation.unwrap_or_default();
        let _run_guard = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(Error::Cancelled),
            guard = self.state.run_lock.lock() => guard,
        };
        self.ensure_loaded().await?;
        if let Some(active) = self.state.session.read().await.active_run.clone() {
            return Err(Error::ActiveRun(active.run_id));
        }

        let run_id = uuid::Uuid::new_v4().to_string();
        let checkpoint = RunCheckpoint::model_call(run_id.clone(), 1, Usage::default());
        self.update_session(|session| {
            session.messages.push(input);
            session.active_run = Some(checkpoint.clone());
            Ok(())
        })
        .await?;

        let sink = options
            .event_sink
            .clone()
            .or_else(|| self.state.event_sink.clone());
        let session_id = self.session_id().await;
        let events = EventContext {
            sink,
            run_id,
            session_id,
            sequence: AtomicU64::new(1),
        };
        events.emit(AgentEventKind::RunStarted);

        let result = self
            .drive_run(checkpoint, &cancellation, &events, false)
            .await;
        emit_run_result(&events, &result);
        result
    }

    /// Resumes an interrupted run from its last safe checkpoint.
    pub async fn resume(&self) -> Result<AgentOutput> {
        self.resume_with(Default::default()).await
    }

    /// Resumes an interrupted run with cancellation and resume options.
    pub async fn resume_with(&self, options: super::ResumeOptions) -> Result<AgentOutput> {
        let cancellation = options.cancellation.unwrap_or_default();
        let _run_guard = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(Error::Cancelled),
            guard = self.state.run_lock.lock() => guard,
        };
        self.ensure_loaded().await?;
        let checkpoint = self
            .state
            .session
            .read()
            .await
            .active_run
            .clone()
            .ok_or(Error::NoActiveRun)?;
        checkpoint.validate()?;

        if checkpoint.tool_started() && !options.retry_uncertain_tools {
            let call = checkpoint.pending_tool_call().ok_or_else(|| {
                Error::Protocol("started tool checkpoint has no pending call".to_owned())
            })?;
            return Err(Error::UnsafeResume {
                tool: call.name.clone(),
                call_id: call.id.clone(),
            });
        }

        let sink = options
            .event_sink
            .clone()
            .or_else(|| self.state.event_sink.clone());
        let events = EventContext {
            sink,
            run_id: checkpoint.run_id.clone(),
            session_id: self.session_id().await,
            sequence: AtomicU64::new(1),
        };
        events.emit(AgentEventKind::RunStarted);
        let result = self
            .drive_run(
                checkpoint,
                &cancellation,
                &events,
                options.retry_uncertain_tools,
            )
            .await;
        emit_run_result(&events, &result);
        result
    }

    /// Resets conversation history while preserving the session ID.
    pub async fn reset(&self) -> Result<()> {
        let _run_guard = self.state.run_lock.lock().await;
        self.ensure_loaded().await?;
        let (id, revision) = {
            let session = self.state.session.read().await;
            (session.id.clone(), session.revision)
        };
        {
            let mut session = self.state.session.write().await;
            *session = crate::Session::with_id(id);
            session.revision = revision.saturating_add(1);
        }
        self.state
            .loaded
            .store(true, std::sync::atomic::Ordering::Release);
        self.persist_session().await
    }

    /// Deletes the persisted session and clears local history.
    ///
    /// The session ID is retained so subsequent runs start a fresh conversation
    /// with the same identity.
    pub async fn delete_session(&self) -> Result<()> {
        let _run_guard = self.state.run_lock.lock().await;
        let id = self.state.session.read().await.id.clone();
        if let Some(store) = &self.state.session_store {
            store.delete(&id).await?;
        }
        *self.state.session.write().await = crate::Session::with_id(id);
        self.state
            .loaded
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    #[tracing::instrument(
        name = "agent.run",
        skip_all,
        fields(
            run_id = %checkpoint.run_id,
            session_id = %events.session_id,
            model = %self.shared.model.name()
        ),
        err
    )]
    async fn drive_run(
        &self,
        mut checkpoint: RunCheckpoint,
        cancellation: &CancellationToken,
        events: &EventContext,
        retry_uncertain_tools: bool,
    ) -> Result<AgentOutput> {
        let mut history = self.state.session.read().await.messages.clone();
        validate_history(&history)?;
        let query = history
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .map(Message::text)
            .unwrap_or_default();
        let session_id = self.session_id().await;
        let memory_context = self.recall_memory(&query, cancellation, events).await?;
        let mut usage = checkpoint.usage;
        let run_id = checkpoint.run_id.clone();

        loop {
            match checkpoint.phase.clone() {
                RunPhase::Tools {
                    assistant,
                    completed_results,
                    ..
                } => {
                    if checkpoint.tool_started() && !retry_uncertain_tools {
                        let call = checkpoint.pending_tool_call().ok_or_else(|| {
                            Error::Protocol(
                                "started tool checkpoint has no pending call".to_owned(),
                            )
                        })?;
                        return Err(Error::UnsafeResume {
                            tool: call.name.clone(),
                            call_id: call.id.clone(),
                        });
                    }

                    let turn = checkpoint.turn;
                    let mut results = completed_results;
                    while results.len() < assistant.tool_calls.len() {
                        check_cancelled(cancellation)?;
                        let call = assistant.tool_calls[results.len()].clone();
                        checkpoint.mark_tool_started();
                        self.persist_checkpoint(&checkpoint).await?;

                        let output = self.execute_tool(turn, &call, cancellation, events).await?;
                        let content = tool_content(&output);
                        let result = Message::tool_result(call.id.clone(), content)
                            .with_name(call.name.clone());
                        checkpoint.push_tool_result(result.clone());
                        self.persist_checkpoint(&checkpoint).await?;
                        events.emit(AgentEventKind::ToolCompleted {
                            turn,
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            output,
                        });
                        results.push(result);
                    }

                    history.push(assistant);
                    history.extend(results);
                    let next_turn = checkpoint.turn.saturating_add(1);
                    if next_turn > self.shared.config.max_turns {
                        return Err(Error::TurnLimit(self.shared.config.max_turns));
                    }
                    checkpoint =
                        RunCheckpoint::model_call(run_id.clone(), next_turn, checkpoint.usage);
                    self.update_session(|session| {
                        session.messages = history.clone();
                        session.active_run = Some(checkpoint.clone());
                        Ok(())
                    })
                    .await?;
                }
                RunPhase::ModelCall => {
                    let turn = checkpoint.turn;
                    if turn > self.shared.config.max_turns {
                        return Err(Error::TurnLimit(self.shared.config.max_turns));
                    }
                    check_cancelled(cancellation)?;
                    events.emit(AgentEventKind::TurnStarted { turn });

                    let fixed_messages = self.fixed_messages(memory_context.as_deref());
                    let tool_specs = self.shared.tools.specs();
                    history = self
                        .prepare_context(
                            history,
                            &fixed_messages,
                            &tool_specs,
                            cancellation,
                            events,
                        )
                        .await?;
                    self.update_session(|session| {
                        session.messages = history.clone();
                        session.active_run = Some(checkpoint.clone());
                        Ok(())
                    })
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
                    let model_span = tracing::info_span!(
                        "model.call",
                        model = %self.shared.model.name(),
                        turn,
                        streaming = self.shared.config.streaming
                    );
                    let response = async {
                        if self.shared.config.streaming {
                            let on_delta = |delta| {
                                events.emit(AgentEventKind::ModelDelta { turn, delta });
                            };
                            cancellable(
                                cancellation,
                                self.shared.model.complete_stream(request, &on_delta),
                            )
                            .await
                        } else {
                            cancellable(cancellation, self.shared.model.complete(request)).await
                        }
                    }
                    .instrument(model_span)
                    .await?;
                    check_cancelled(cancellation)?;
                    validate_response(&response.message)?;

                    usage += response.usage;
                    tracing::debug!(
                        input_tokens = response.usage.input_tokens,
                        output_tokens = response.usage.output_tokens,
                        finish_reason = ?response.finish_reason,
                        "model call completed"
                    );
                    events.emit(AgentEventKind::ModelCompleted {
                        turn,
                        usage: response.usage,
                        finish_reason: response.finish_reason.clone(),
                    });

                    let assistant = response.message;
                    if assistant.tool_calls.is_empty() {
                        history.push(assistant.clone());
                        let output = AgentOutput {
                            run_id: events.run_id.clone(),
                            session_id: session_id.clone(),
                            text: assistant.text(),
                            message: assistant,
                            turns: turn,
                            usage,
                            finish_reason: response.finish_reason,
                        };
                        self.update_session(|session| {
                            session.messages = history.clone();
                            session.active_run = None;
                            Ok(())
                        })
                        .await?;
                        return Ok(output);
                    }

                    checkpoint = RunCheckpoint::tools(run_id.clone(), turn, usage, assistant);
                    self.update_session(|session| {
                        session.messages = history.clone();
                        session.active_run = Some(checkpoint.clone());
                        Ok(())
                    })
                    .await?;
                }
            }
        }
    }

    #[tracing::instrument(
        name = "tool.call",
        skip_all,
        fields(
            turn,
            tool = %call.name,
            call_id = %call.id,
            timeout_ms = ?self.shared.config.tool_timeout_ms,
            timed_out = tracing::field::Empty,
            truncated = tracing::field::Empty,
            output_chars = tracing::field::Empty
        ),
        err
    )]
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
        let mut timed_out = false;
        let result = if let Some(timeout_ms) = self.shared.config.tool_timeout_ms {
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(Error::Cancelled),
                result = tokio::time::timeout(
                    Duration::from_millis(timeout_ms),
                    tool.call(call.arguments.clone(), &context),
                ) => match result {
                    Ok(result) => result,
                    Err(_) => {
                        timed_out = true;
                        Err(Error::ToolTimeout {
                            tool: call.name.clone(),
                            timeout_ms,
                        })
                    }
                }
            }
        } else {
            cancellable(cancellation, tool.call(call.arguments.clone(), &context)).await
        };
        tracing::Span::current().record("timed_out", timed_out);
        let output = match result {
            Ok(output) => output,
            Err(error) if error.is_cancelled() || cancellation.is_cancelled() => {
                return Err(Error::Cancelled);
            }
            Err(error) if self.shared.config.fail_on_tool_error => return Err(error),
            Err(error) => ToolOutput::error(error.to_string()),
        };
        let (output, truncated) =
            truncate_tool_output(output, self.shared.config.max_tool_output_chars);
        tracing::Span::current().record("truncated", truncated);
        tracing::Span::current().record("output_chars", output.content.chars().count());
        Ok(output)
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

        let compression_span = tracing::info_span!(
            "context.compress",
            budget_tokens = available,
            history_tokens = current
        );
        let compression = async {
            cancellable(
                cancellation,
                compressor.compress(CompressionRequest {
                    messages: history,
                    budget_tokens: available,
                    token_counter: self.shared.token_counter.clone(),
                }),
            )
            .await
        }
        .instrument(compression_span)
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
        tracing::debug!(
            before_tokens = compression.before_tokens,
            after_tokens = compression.after_tokens,
            removed_messages = compression.removed_messages,
            summarized = compression.summarized,
            "context compressed"
        );

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

        let span = tracing::info_span!(
            "memory.recall",
            limit = self.shared.memory_options.recall_limit
        );
        let entries = async {
            cancellable(
                cancellation,
                memory.recall(
                    &MemoryQuery::new(query).with_limit(self.shared.memory_options.recall_limit),
                ),
            )
            .await
        }
        .instrument(span)
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

        tracing::debug!(count = injected, "memory context injected");
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
        let span = tracing::info_span!("session.load", session_id = %id);
        async {
            if let Some(store) = &self.state.session_store
                && let Some(session) = store.load(&id).await?
            {
                if session.id != id {
                    return Err(Error::Protocol(format!(
                        "session store returned session `{}` for requested id `{id}`",
                        session.id
                    )));
                }
                if session.schema_version != SESSION_SCHEMA_VERSION {
                    return Err(Error::UnsupportedSessionVersion {
                        found: session.schema_version,
                        supported: SESSION_SCHEMA_VERSION,
                    });
                }
                if let Some(checkpoint) = &session.active_run {
                    checkpoint.validate()?;
                }
                *self.state.session.write().await = session;
            }
            Ok::<(), Error>(())
        }
        .instrument(span)
        .await?;
        self.state.loaded.store(true, Ordering::Release);
        Ok(())
    }

    async fn persist_checkpoint(&self, checkpoint: &RunCheckpoint) -> Result<()> {
        self.update_session(|session| {
            session.active_run = Some(checkpoint.clone());
            Ok(())
        })
        .await
        .map(|_| ())
    }

    async fn update_session(
        &self,
        update: impl FnOnce(&mut crate::Session) -> Result<()>,
    ) -> Result<crate::Session> {
        let mut session = self.state.session.read().await.clone();
        update(&mut session)?;
        session.touch();
        let span = tracing::info_span!(
            "session.save",
            session_id = %session.id,
            revision = session.revision,
            active_run = session.active_run.is_some(),
            messages = session.messages.len()
        );
        async {
            if let Some(store) = &self.state.session_store {
                store.save(&session).await?;
            }
            Ok::<(), Error>(())
        }
        .instrument(span)
        .await?;
        *self.state.session.write().await = session.clone();
        Ok(session)
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
        let span = tracing::info_span!(
            "session.save",
            session_id = %session.id,
            revision = session.revision,
            active_run = session.active_run.is_some(),
            messages = session.messages.len()
        );
        async { store.save(&session).await }.instrument(span).await
    }
}

fn tool_content(output: &ToolOutput) -> String {
    match (output.content.is_empty(), output.is_error) {
        (true, _) => "(tool returned no output)".to_owned(),
        (false, true) => format!("Tool error: {}", output.content),
        (false, false) => output.content.clone(),
    }
}

fn truncate_tool_output(mut output: ToolOutput, max_chars: Option<usize>) -> (ToolOutput, bool) {
    let Some(max_chars) = max_chars else {
        return (output, false);
    };
    let total_chars = output.content.chars().count();
    if total_chars <= max_chars {
        return (output, false);
    }
    let removed = total_chars.saturating_sub(max_chars);
    let mut content = output.content.chars().take(max_chars).collect::<String>();
    content.push_str(&format!("\n...[truncated {removed} chars]"));
    output.content = content;
    (output, true)
}

fn emit_run_result(events: &EventContext, result: &Result<AgentOutput>) {
    match result {
        Ok(output) => {
            tracing::info!(
                run_id = %output.run_id,
                turns = output.turns,
                input_tokens = output.usage.input_tokens,
                output_tokens = output.usage.output_tokens,
                "agent run completed"
            );
            events.emit(AgentEventKind::RunCompleted {
                text: output.text.clone(),
                turns: output.turns,
                usage: output.usage,
            });
        }
        Err(error) => events.emit(AgentEventKind::RunFailed {
            error: error.to_string(),
        }),
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
