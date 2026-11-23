//! Optional SKILL.md discovery and prompt integration.
//!
//! Skills are intentionally not part of the agent loop. The registry turns a
//! directory of `SKILL.md` files into a prompt catalog plus a `read_skill` tool.

mod loader;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{AgentBuilder, Error, Result, Tool, ToolContext, ToolOutput, ToolSpec};

/// One parsed skill.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Skill identifier from frontmatter, or its directory name.
    pub name: String,
    /// Short applicability description.
    pub description: String,
    /// Full instruction body from `SKILL.md`.
    pub instructions: String,
    /// Source file.
    pub path: PathBuf,
    /// Additional flat frontmatter values.
    pub metadata: BTreeMap<String, String>,
}

/// A set of discovered skills.
#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    skills: BTreeMap<String, Arc<Skill>>,
}

impl SkillRegistry {
    /// Discovers every `SKILL.md` under `root` up to four directory levels.
    pub fn load_dir(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_dir_with_depth(root, 4)
    }

    /// Discovers skills under `root` with an explicit directory depth limit.
    pub fn load_dir_with_depth(root: impl AsRef<Path>, max_depth: usize) -> Result<Self> {
        let paths = loader::discover(root.as_ref(), max_depth)?;
        let mut registry = Self::default();
        for path in paths {
            registry.insert(loader::load_skill(&path)?)?;
        }
        Ok(registry)
    }

    /// Loads exactly one `SKILL.md` file.
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut registry = Self::default();
        registry.insert(loader::load_skill(path.as_ref())?)?;
        Ok(registry)
    }

    /// Builds a registry from already parsed skills.
    pub fn from_skills(skills: impl IntoIterator<Item = Skill>) -> Result<Self> {
        let mut registry = Self::default();
        for skill in skills {
            registry.insert(skill)?;
        }
        Ok(registry)
    }

    /// Adds one skill, rejecting duplicate names.
    pub fn insert(&mut self, skill: Skill) -> Result<()> {
        if skill.name.trim().is_empty() {
            return Err(Error::Configuration(format!(
                "skill at {} has an empty name",
                skill.path.display()
            )));
        }
        if self.skills.contains_key(&skill.name) {
            return Err(Error::Configuration(format!(
                "duplicate skill name `{}`",
                skill.name
            )));
        }
        self.skills.insert(skill.name.clone(), Arc::new(skill));
        Ok(())
    }

    /// Looks up a skill by name.
    pub fn get(&self, name: &str) -> Option<&Arc<Skill>> {
        self.skills.get(name)
    }

    /// Returns skills in name order.
    pub fn skills(&self) -> Vec<Arc<Skill>> {
        self.skills.values().cloned().collect()
    }

    /// Number of skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Returns whether no skills are loaded.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Renders the catalog intended for a system prompt section.
    pub fn catalog_prompt(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }
        let mut prompt = String::from(
            "## Available skills\n\
             Skills are reusable instruction packs. When a task clearly matches \
             one, call `read_skill` before proceeding.\n",
        );
        for skill in self.skills.values() {
            prompt.push_str(&format!("- `{}`: {}\n", skill.name, skill.description));
        }
        prompt
    }

    /// Creates the `read_skill` tool.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![Arc::new(SkillTool::new(self.clone()))]
    }

    /// Adds the catalog section and `read_skill` tool to a builder.
    pub fn install(&self, builder: AgentBuilder) -> AgentBuilder {
        if self.is_empty() {
            return builder;
        }
        let catalog = self.catalog_prompt();
        let builder = if catalog.is_empty() {
            builder
        } else {
            builder.prompt_section(catalog)
        };
        builder.tools(self.tools())
    }
}

/// Loads full skill instructions on demand.
pub struct SkillTool {
    registry: SkillRegistry,
}

impl SkillTool {
    /// Creates the tool.
    pub fn new(registry: SkillRegistry) -> Self {
        Self { registry }
    }
}

#[derive(Deserialize)]
struct ReadSkillArgs {
    name: String,
}

#[async_trait]
impl Tool for SkillTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "read_skill",
            "Load the full instructions for one available skill. Use this when a task matches a skill in the skill catalog.",
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name from the catalog."
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
        )
    }

    async fn call(&self, arguments: Value, _context: &ToolContext) -> Result<ToolOutput> {
        let args: ReadSkillArgs = serde_json::from_value(arguments)?;
        let Some(skill) = self.registry.get(args.name.trim()) else {
            let mut names = self
                .registry
                .skills()
                .iter()
                .map(|skill| skill.name.clone())
                .collect::<Vec<_>>()
                .join(", ");
            if names.is_empty() {
                names = "(none)".to_owned();
            }
            return Ok(ToolOutput::error(format!(
                "unknown skill; available skills: {names}"
            )));
        };

        let mut output = format!("# Skill: {}\n\n", skill.name);
        if !skill.description.is_empty() {
            output.push_str(&skill.description);
            output.push_str("\n\n");
        }
        output.push_str(&skill.instructions);
        Ok(ToolOutput::new(output))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn discovers_and_parses_skills() {
        let root = tempdir().unwrap();
        let skill_dir = root.path().join("writer");
        fs::create_dir(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: writer\ndescription: Write crisp product copy\nowner: docs\n---\n# Writer\nPrefer short sentences.\n",
        )
        .unwrap();

        let registry = SkillRegistry::load_dir(root.path()).unwrap();
        assert_eq!(registry.len(), 1);
        let skill = registry.get("writer").unwrap();
        assert_eq!(skill.description, "Write crisp product copy");
        assert!(skill.instructions.contains("Prefer short sentences"));
        assert!(registry.catalog_prompt().contains("`writer`"));
    }

    #[test]
    fn falls_back_to_directory_name() {
        let root = tempdir().unwrap();
        let skill_dir = root.path().join("summarizer");
        fs::create_dir(&skill_dir).unwrap();
        fs::write(skill_dir.join("SKILL.md"), "Summarize carefully.").unwrap();

        let registry = SkillRegistry::load_dir(root.path()).unwrap();
        assert!(registry.get("summarizer").is_some());
        assert_eq!(
            registry.get("summarizer").unwrap().instructions,
            "Summarize carefully."
        );
    }
}
