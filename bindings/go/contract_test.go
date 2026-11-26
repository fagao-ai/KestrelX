package kestrel

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestContractVersionAndCapabilities(t *testing.T) {
	if ContractVersion() != 1 {
		t.Fatalf("contract version = %d", ContractVersion())
	}
	capabilities, err := Capabilities()
	if err != nil {
		t.Fatal(err)
	}
	for _, capability := range []string{
		"mcp_http",
		"openai_responses",
		"anthropic",
		"gemini",
	} {
		if !slices.Contains(capabilities, capability) {
			t.Fatalf("missing %s in %v", capability, capabilities)
		}
	}
}

func TestSharedFixtureValidation(t *testing.T) {
	fixture, err := os.ReadFile(
		filepath.Join("..", "..", "kestrel-contract", "fixtures", "full.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateConfig(fixture, true); err != nil {
		t.Fatal(err)
	}

	providerFixture, err := os.ReadFile(
		filepath.Join("..", "..", "kestrel-contract", "fixtures", "provider.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateConfig(providerFixture, false); err != nil {
		t.Fatal(err)
	}
}
