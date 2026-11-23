use std::sync::{Arc, RwLock};

use kestrel::{AgentEvent, EventSink};

use crate::callbacks::EventCallback;

#[derive(Clone, Default)]
pub(crate) struct EventSlot {
    callback: Arc<RwLock<Option<EventCallback>>>,
}

impl EventSlot {
    pub(crate) fn set(&self, callback: Option<EventCallback>) {
        *self
            .callback
            .write()
            .unwrap_or_else(|error| error.into_inner()) = callback;
    }
}

pub(crate) struct CallbackEventSink {
    slot: EventSlot,
}

impl CallbackEventSink {
    pub(crate) fn new(slot: EventSlot) -> Self {
        Self { slot }
    }
}

impl EventSink for CallbackEventSink {
    fn emit(&self, event: AgentEvent) {
        let callback = self
            .slot
            .callback
            .read()
            .unwrap_or_else(|error| error.into_inner());
        let Some(callback) = *callback else {
            return;
        };
        if let Ok(event_json) = serde_json::to_string(&event) {
            let _ = callback.invoke(&event_json);
        }
    }
}
