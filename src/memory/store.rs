use std::{
    collections::{BTreeMap, HashSet},
    sync::{Arc, RwLock},
};

use async_trait::async_trait;

use crate::{Memory, MemoryEntry, MemoryQuery, Result};

/// A process-local long-term memory with keyword and tag retrieval.
#[derive(Debug, Clone)]
pub struct InMemoryMemory {
    entries: Arc<RwLock<BTreeMap<String, MemoryEntry>>>,
    capacity: Option<usize>,
}

impl Default for InMemoryMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryMemory {
    /// Creates an unbounded memory.
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(BTreeMap::new())),
            capacity: None,
        }
    }

    /// Creates a memory that evicts the oldest entries when full.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(BTreeMap::new())),
            capacity: Some(capacity.max(1)),
        }
    }

    /// Returns the number of stored entries.
    pub fn len(&self) -> usize {
        self.read().len()
    }

    /// Returns whether the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<String, MemoryEntry>> {
        self.entries
            .read()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<String, MemoryEntry>> {
        self.entries
            .write()
            .unwrap_or_else(|error| error.into_inner())
    }
}

#[async_trait]
impl Memory for InMemoryMemory {
    async fn remember(&self, entry: MemoryEntry) -> Result<()> {
        let mut entries = self.write();
        entries.insert(entry.id.clone(), entry);

        if let Some(capacity) = self.capacity {
            while entries.len() > capacity {
                let oldest = entries
                    .values()
                    .min_by_key(|entry| (entry.created_at_ms, entry.id.clone()))
                    .map(|entry| entry.id.clone());
                if let Some(id) = oldest {
                    entries.remove(&id);
                } else {
                    break;
                }
            }
        }
        Ok(())
    }

    async fn recall(&self, query: &MemoryQuery) -> Result<Vec<MemoryEntry>> {
        if query.limit == 0 {
            return Ok(Vec::new());
        }

        let normalized_query = query.text.trim().to_lowercase();
        let query_terms = normalized_query
            .split_whitespace()
            .filter(|term| !term.is_empty())
            .collect::<Vec<_>>();

        let mut matches = self
            .read()
            .values()
            .filter_map(|entry| {
                let tags_match = query.tags.is_empty()
                    || query.tags.iter().any(|tag| {
                        entry
                            .tags
                            .iter()
                            .any(|entry_tag| entry_tag.eq_ignore_ascii_case(tag))
                    });
                if !tags_match {
                    return None;
                }

                let text = entry.text.to_lowercase();
                let mut score = 0usize;
                if normalized_query.is_empty() {
                    score = 1;
                } else if text.contains(&normalized_query) {
                    score += 100;
                }
                score += query_terms
                    .iter()
                    .filter(|term| text.contains(**term))
                    .count()
                    * 5;
                score += query
                    .tags
                    .iter()
                    .filter(|tag| {
                        entry
                            .tags
                            .iter()
                            .any(|entry_tag| entry_tag.eq_ignore_ascii_case(tag))
                    })
                    .count()
                    * 20;

                if !normalized_query.is_ascii() {
                    let query_characters = normalized_query
                        .chars()
                        .filter(|character| character.is_alphanumeric())
                        .collect::<HashSet<_>>();
                    let overlap = query_characters
                        .iter()
                        .filter(|character| text.contains(**character))
                        .count();
                    if overlap >= 2 || (query_characters.len() == 1 && overlap == 1) {
                        score += overlap * 3;
                    }
                }

                (score > 0).then_some((score, entry.clone()))
            })
            .collect::<Vec<_>>();

        matches.sort_by(|(left_score, left), (right_score, right)| {
            right_score
                .cmp(left_score)
                .then_with(|| right.created_at_ms.cmp(&left.created_at_ms))
                .then_with(|| right.id.cmp(&left.id))
        });
        Ok(matches
            .into_iter()
            .take(query.limit)
            .map(|(_, entry)| entry)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn recalls_by_keyword_and_tag() {
        let memory = InMemoryMemory::new();
        memory
            .remember(MemoryEntry::new("The launch codename is Kestrel").with_tags(["launch"]))
            .await
            .unwrap();
        memory
            .remember(MemoryEntry::new("The office is in Shanghai"))
            .await
            .unwrap();

        let results = memory
            .recall(&MemoryQuery::new("launch codename"))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("Kestrel"));

        let results = memory
            .recall(&MemoryQuery {
                text: String::new(),
                tags: vec!["launch".to_owned()],
                limit: 5,
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn recalls_chinese_by_character_overlap() {
        let memory = InMemoryMemory::new();
        memory
            .remember(MemoryEntry::new("项目上线时间是六月"))
            .await
            .unwrap();

        let results = memory
            .recall(&MemoryQuery::new("项目什么时候上线"))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }
}
