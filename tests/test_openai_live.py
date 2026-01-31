from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_openai_gpt5_mini_live():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    from openai import OpenAI

    client = OpenAI()
    resp = client.responses.create(
        model="gpt-5-mini",
        input="Reply with the single word OK.",
    )
    output = resp.output_text if hasattr(resp, "output_text") else ""
    assert output.strip() != ""
