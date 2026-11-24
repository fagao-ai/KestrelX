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
	if !slices.Contains(capabilities, "mcp_http") {
		t.Fatalf("missing mcp_http in %v", capabilities)
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
}
