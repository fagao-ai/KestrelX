package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	kestrel "kestrel-go"
)

func main() {
	var mu sync.Mutex
	modelCalls := 0
	model := func(request kestrel.ModelRequest, emit func(kestrel.ModelDelta)) (kestrel.ModelResponse, error) {
		mu.Lock()
		modelCalls++
		call := modelCalls
		mu.Unlock()
		if call == 1 {
			emit(kestrel.ModelDelta{Type: "text", Text: "calling Go tool\n"})
			return kestrel.ModelResponse{
				Message: kestrel.Message{
					Role: kestrel.RoleAssistant,
					ToolCalls: []kestrel.ToolCall{{
						ID:   "go-call-1",
						Name: "echo",
						Arguments: map[string]any{
							"text": "hello from Go",
						},
					}},
				},
				FinishReason: "tool_calls",
			}, nil
		}
		result := ""
		for _, message := range request.Messages {
			if message.Role == kestrel.RoleTool {
				result = message.Text()
			}
		}
		return kestrel.ModelResponse{
			Message: kestrel.Message{
				Role:    kestrel.RoleAssistant,
				Content: []kestrel.ContentPart{kestrel.TextPart("tool said: " + result)},
			},
			FinishReason: "stop",
		}, nil
	}

	agent, err := kestrel.New(
		map[string]any{"streaming": true},
		kestrel.WithStreamingModel(model),
		kestrel.WithEventHandler(func(event kestrel.Event) {
			if event.Type == "model_delta" && event.Delta != nil {
				fmt.Print(event.Delta.Text)
			}
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer agent.Close()

	err = agent.RegisterTool(
		kestrel.ToolSpec{
			Name:        "echo",
			Description: "Echo supplied text",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
				},
				"required": []string{"text"},
			},
		},
		func(_ string, arguments map[string]any) (kestrel.ToolOutput, error) {
			return kestrel.ToolOutput{Content: "echo:" + arguments["text"].(string)}, nil
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	output, err := agent.Run(ctx, "use the echo tool")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(output.Text)
}
