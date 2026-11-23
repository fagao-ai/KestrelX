use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    Agent, AgentTemplate, Message, Result, RunOptions, Tool, ToolContext, ToolOutput, ToolSpec,
};

/// Exposes an isolated child agent as a tool.
///
/// Each invocation gets a fresh session. The child shares the parent's model,
/// tools, memory, prompts, and compressor but has an independent run lock.
pub struct SubAgentTool {
    name: String,
    description: String,
    template: AgentTemplate,
    max_depth: usize,
}

impl SubAgentTool {
    /// Creates a sub-agent tool from an agent template.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        template: AgentTemplate,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            template,
            max_depth: 2,
        }
    }

    /// Creates a sub-agent tool from an existing agent.
    pub fn from_agent(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: &Agent,
    ) -> Self {
        Self::new(name, description, agent.template())
    }

    /// Overrides the maximum nesting depth.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }
}

#[derive(Deserialize)]
struct DelegateArgs {
    task: String,
    context: Option<String>,
}

#[async_trait]
impl Tool for SubAgentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            &self.name,
            &self.description,
            json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A self-contained task for the sub-agent."
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context the sub-agent cannot infer."
                    }
                },
                "required": ["task"],
                "additionalProperties": false
            }),
        )
    }

    async fn call(&self, arguments: Value, context: &ToolContext) -> Result<ToolOutput> {
        if context.depth >= self.max_depth {
            return Ok(ToolOutput::error(format!(
                "sub-agent depth limit ({}) reached",
                self.max_depth
            )));
        }

        let args: DelegateArgs = serde_json::from_value(arguments)?;
        if args.task.trim().is_empty() {
            return Ok(ToolOutput::error("sub-agent task cannot be empty"));
        }
        let mut prompt = args.task.trim().to_owned();
        if let Some(additional) = args.context.filter(|value| !value.trim().is_empty()) {
            prompt.push_str("\n\nAdditional context:\n");
            prompt.push_str(additional.trim());
        }

        let child = self
            .template
            .spawn_at_depth(context.depth.saturating_add(1));
        let output = child
            .run_with(
                Message::user(prompt),
                RunOptions {
                    cancellation: Some(context.cancellation()),
                    event_sink: None,
                },
            )
            .await?;
        if output.text.trim().is_empty() {
            Ok(ToolOutput::new(
                "Sub-agent completed without returning text.",
            ))
        } else {
            Ok(ToolOutput::new(output.text))
        }
    }
}
