"""Contract conformance smoke tests for the Python binding."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import kestrel


class ContractTests(unittest.TestCase):
    def test_version_and_capabilities(self) -> None:
        self.assertEqual(kestrel.contract_version(), 1)
        capabilities = kestrel.capabilities()
        self.assertIn("streaming", capabilities)
        self.assertIn("mcp_http", capabilities)

    def test_shared_fixture(self) -> None:
        fixture = (
            Path(__file__).resolve().parents[2]
            / "kestrel-contract"
            / "fixtures"
            / "full.json"
        )
        kestrel.validate_config(json.loads(fixture.read_text()), True)


if __name__ == "__main__":
    unittest.main()
