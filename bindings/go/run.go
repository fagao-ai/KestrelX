package kestrel

/*
#include "kestrel.h"
*/
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"
)

// ErrRunClosed is returned when a run handle is used after Close.
var ErrRunClosed = errors.New("Kestrel run is closed")

// Run is a pollable handle for one asynchronous run.
type Run struct {
	mu     sync.Mutex
	owner  *Agent
	ptr    *C.KestrelRun
	closed bool
}

// NextEvent waits for the next event. It returns io.EOF when the run ends.
func (r *Run) NextEvent(ctx context.Context) (*Event, error) {
	for {
		if err := ctx.Err(); err != nil {
			r.Cancel()
			return nil, err
		}
		r.mu.Lock()
		if r.closed {
			r.mu.Unlock()
			return nil, ErrRunClosed
		}
		var eventC *C.char
		var errorC *C.char
		status := C.kestrel_run_next_event(r.ptr, 100, &eventC, &errorC)
		r.mu.Unlock()
		switch status {
		case 1:
			defer C.kestrel_string_free(eventC)
			event, err := parseEvent([]byte(C.GoString(eventC)))
			if err != nil {
				return nil, err
			}
			return &event, nil
		case 0:
			continue
		case 2:
			return nil, io.EOF
		default:
			return nil, takeCError(errorC)
		}
	}
}

// Wait waits for final output.
func (r *Run) Wait(ctx context.Context) (*AgentOutput, error) {
	for {
		if err := ctx.Err(); err != nil {
			r.Cancel()
			return nil, err
		}
		r.mu.Lock()
		if r.closed {
			r.mu.Unlock()
			return nil, ErrRunClosed
		}
		var outputC *C.char
		var errorC *C.char
		status := C.kestrel_run_wait(r.ptr, 100, &outputC, &errorC)
		r.mu.Unlock()
		switch status {
		case 1:
			defer C.kestrel_string_free(outputC)
			var output AgentOutput
			if err := json.Unmarshal([]byte(C.GoString(outputC)), &output); err != nil {
				return nil, err
			}
			return &output, nil
		case 0:
			continue
		default:
			return nil, takeCError(errorC)
		}
	}
}

// Cancel requests cancellation without freeing the run.
func (r *Run) Cancel() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.closed {
		C.kestrel_run_cancel(r.ptr)
	}
}

// Close cancels and frees the run, waiting for in-flight callbacks.
func (r *Run) Close() {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}
	r.closed = true
	owner := r.owner
	C.kestrel_run_free(r.ptr)
	r.mu.Unlock()
	if owner != nil {
		owner.removeRun(r)
	}
}
