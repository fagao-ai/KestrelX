use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    path::{Path, PathBuf},
};

use crate::{Error, Result};

use super::Skill;

pub(super) fn discover(root: &Path, max_depth: usize) -> Result<Vec<PathBuf>> {
    let metadata = fs::metadata(root).map_err(|error| {
        Error::Configuration(format!(
            "cannot read skills path {}: {error}",
            root.display()
        ))
    })?;
    if metadata.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    if !metadata.is_dir() {
        return Err(Error::Configuration(format!(
            "skills path {} is neither a file nor a directory",
            root.display()
        )));
    }

    let mut skills = Vec::new();
    let mut queue = VecDeque::from([(root.to_path_buf(), 0usize)]);
    while let Some((directory, depth)) = queue.pop_front() {
        let mut entries = fs::read_dir(&directory)?.collect::<std::result::Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());

        for entry in entries {
            let file_type = entry.file_type()?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if file_type.is_file() && name.eq_ignore_ascii_case("SKILL.md") {
                skills.push(entry.path());
                continue;
            }
            if file_type.is_dir()
                && depth < max_depth
                && !matches!(name.as_ref(), ".git" | "node_modules" | "target")
            {
                queue.push_back((entry.path(), depth + 1));
            }
        }
    }
    Ok(skills)
}

pub(super) fn load_skill(path: &Path) -> Result<Skill> {
    let source = fs::read_to_string(path).map_err(|error| {
        Error::Configuration(format!("cannot read skill {}: {error}", path.display()))
    })?;
    let (mut metadata, instructions) = parse_frontmatter(&source)?;
    let fallback_name = path
        .parent()
        .and_then(Path::file_name)
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "skill".to_owned());
    let name = metadata
        .remove("name")
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(fallback_name);
    let description = metadata.remove("description").unwrap_or_default();

    Ok(Skill {
        name,
        description,
        instructions: instructions.trim().to_owned(),
        path: path.to_path_buf(),
        metadata,
    })
}

fn parse_frontmatter(source: &str) -> Result<(BTreeMap<String, String>, &str)> {
    let source = source.strip_prefix('\u{feff}').unwrap_or(source);
    let Some(rest) = source
        .strip_prefix("---\n")
        .or_else(|| source.strip_prefix("---\r\n"))
    else {
        return Ok((BTreeMap::new(), source));
    };

    let mut offset = 0usize;
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" || trimmed == "..." {
            let body = &rest[offset + line.len()..];
            return Ok((parse_values(&rest[..offset])?, body));
        }
        offset += line.len();
    }

    Err(Error::Configuration(
        "SKILL.md frontmatter is missing a closing `---`".to_owned(),
    ))
}

fn parse_values(frontmatter: &str) -> Result<BTreeMap<String, String>> {
    let mut values = BTreeMap::new();
    for (index, line) in frontmatter.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            return Err(Error::Configuration(format!(
                "invalid SKILL.md frontmatter line {}: expected `key: value`",
                index + 1
            )));
        };
        let key = key.trim();
        if key.is_empty() {
            return Err(Error::Configuration(format!(
                "invalid SKILL.md frontmatter line {}: empty key",
                index + 1
            )));
        }
        values.insert(key.to_owned(), unquote(value.trim()));
    }
    Ok(values)
}

fn unquote(value: &str) -> String {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        let first = bytes[0];
        let last = bytes[value.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return value[1..value.len() - 1].to_owned();
        }
    }
    value.to_owned()
}
