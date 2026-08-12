from __future__ import annotations

import torch

from icalens import AnalysisResult
from icalens.html import analysis_iframe, write_explorer_html


def analysis_result() -> AnalysisResult:
    return AnalysisResult(
        tokens=("Ġhello",),
        token_texts=(" hello",),
        token_labels=(" hello",),
        token_tooltips=("Ġhello",),
        token_groups=(),
        token_ids=torch.tensor([1]),
        positions=torch.tensor([3]),
        activations=torch.tensor([[1.0, 2.0]]),
        scores=torch.tensor([[-2.0, 1.0]]),
        energy=torch.tensor([[0.8, 0.2]]),
        model="example/model@abc",
        layer=6,
        input_text="hello",
        token_scope="all text tokens",
        messages=(),
        component_profiles={
            0: {
                "dominant_sign": "negative",
                "sign_statistics": {
                    "positive_fraction": 0.4,
                    "negative_fraction": 0.6,
                    "positive_energy_fraction": 0.2,
                    "negative_energy_fraction": 0.8,
                },
                "occurrences": [
                    {
                        "text": " biology",
                        "context": "interested in biology",
                        "score": -12.0,
                        "energy": 0.4,
                    }
                ],
                "logit_tokens": [{"text": " biology", "logit": 6.5}],
            }
        },
    )


def test_write_explorer_html_is_self_contained_and_escapes_payload(tmp_path) -> None:
    destination = write_explorer_html(
        tmp_path / "nested" / "report.html",
        title="Test explorer",
        model="example/model@abc",
        layer=6,
        input_text="hello </script><script>alert(1)</script>",
        token_scope="assistant",
        messages=[
            {"role": "user", "content": "First turn"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second turn"},
            {"role": "assistant", "content": "Second response"},
        ],
        token_groups=[
            {
                "title": "Prompt template",
                "open": True,
                "tokens": [
                    {
                        "position": 0,
                        "token": "<|user|>",
                        "token_text": "<|user|>",
                        "top": [{"component": 7, "score": 2.5}],
                    }
                ],
            },
        ],
        tokens=[
            {
                "position": 3,
                "token": "Ġhello",
                "token_text": " hello",
                "top": [{"component": 2, "score": -1.25}],
            }
        ],
    )
    html = destination.read_text(encoding="utf-8")
    assert destination.is_file()
    assert "ICA Lens Explorer" in html
    assert "example/model@abc" in html
    assert "hello </script><script>" not in html
    assert "\\u003c/script>" in html
    assert 'selected ? color(component) : "#e7ebf1"' in html
    assert "Prompt template" in html
    assert "<|user|>" not in html
    assert "\\u003c|user|>" in html
    assert "details.open = Boolean(group.open)" in html
    assert "group.tokens.map(token => tokenCard(token, threshold))" in html
    assert '"metric":"score"' in html
    assert 'state.metric === "energy"' in html
    assert 'step="20" value="100"' in html
    assert 'value="3"' in html
    assert '<details class="panel analysis-context" id="analysisContext">' in html
    assert "<summary>Analysis details</summary>" in html
    assert 'document.getElementById("analysisContext").open = !data.compact' in html
    assert (
        'document.querySelector("main").append(document.getElementById("analysisContext"))' in html
    )
    assert 'class="token-position"' not in html
    assert 'content: "Analyzed text"' not in html
    assert "ui-monospace, SFMono-Regular" in html
    assert "white-space: nowrap; text-overflow: ellipsis" in html
    assert 'id="metricSelect"' in html
    assert 'id="topK"' in html
    assert "modelChip.title = data.model" in html
    assert '<span class="chip-label">Model:</span>' in html
    assert '<span class="chip-label">Token scope:</span>' in html
    assert '<span class="chip-label">Tokens:</span>' in html
    assert '"all text tokens": "entire input"' in html
    assert '<div class="context-label" id="inputLabel">Analyzed text</div>' in html
    assert '<blockquote class="input" id="input"></blockquote>' in html
    assert 'textContent = "Analyzed conversation"' in html
    assert "visible in ${visible}/${tokens.length} tokens" in html
    assert "main.getBoundingClientRect().bottom" in html
    assert "const height = Math.max(contentHeight + 4, 180)" in html
    assert 'replace(/\\r\\n|\\r|\\n/g, "↵")' in html
    assert "max-width: 1680px" in html
    assert "`${data.result_group_title} (${data.tokens.length} tokens)`" in html
    assert '<details class="panel" id="resultGroup" open>' in html
    assert '"content":"Second turn"' in html
    assert '<span class="chip-label">User turns:</span>' in html
    assert "card.className = `message ${role}`" in html
    assert html.index('<section class="panel controls">') < html.index('<div id="tokenGroups">')


def test_write_explorer_html_accepts_energy_metric(tmp_path) -> None:
    destination = write_explorer_html(
        tmp_path / "energy.html",
        title="Energy",
        model="example/model",
        layer=1,
        input_text="hello",
        token_scope="all",
        metric="energy",
        tokens=[],
    )

    assert '"metric":"energy"' in destination.read_text(encoding="utf-8")


def test_analysis_result_writes_html(tmp_path) -> None:
    result = analysis_result()

    destination = result.to_html(tmp_path / "analysis.html", metric="energy", top_k=1)
    html = destination.read_text(encoding="utf-8")
    assert '"model":"example/model@abc"' in html
    assert '"metric":"energy"' in html
    assert '"token_text":" hello"' in html
    assert '"token_label":" hello"' in html
    assert '"metrics":{"score":[-2.0,1.0]' in html
    assert '"top_k":1' in html
    assert '"dynamic_metrics":true' in html
    assert '"dominant_sign":"negative"' in html
    assert "Top occurrences:" in html
    assert "Top logit-lens tokens:" in html
    assert 'title="${esc(tooltip)}"' in html
    assert 'replace(/\\r\\n|\\r|\\n/g, "↵")' in html
    assert 'return lines.join("\\n")' in html
    assert '<details class="panel" id="componentProfile" hidden>' in html
    assert "Component profile — C${component} · dominant ${profile.dominant_sign}" in html
    assert "High-energy occurrences · ${profile.dominant_sign}" in html
    assert "Logit-lens tokens · ${profile.dominant_sign}" in html
    assert "Suppressed logit-lens tokens" not in html
    assert "profilePanel.hidden = true" in html
    assert "panel.hidden = false" in html
    assert 'class="profile-target"' in html
    assert "const index = target ? context.indexOf(target) : -1" in html
    assert "highlightedContext(item)" in html


def test_analysis_result_has_notebook_representation() -> None:
    rendered = analysis_result()._repr_html_()
    assert rendered.startswith('<iframe title="ICA Lens Explorer"')
    assert 'sandbox="allow-scripts allow-same-origin"' in rendered
    assert 'scrolling="no"' in rendered
    assert "overflow:hidden" in rendered
    assert "height:720px" in rendered
    assert "example/model@abc" in rendered
    assert "&lt;!doctype html&gt;" in rendered
    assert "&quot;compact&quot;:true" in rendered


def test_analysis_iframe_validates_options() -> None:
    result = analysis_result()
    rendered = analysis_iframe(result, metric="energy", top_k=1, height=400)
    assert "height:400px" in rendered
    assert "energy" in rendered


def test_conversation_result_uses_message_token_groups() -> None:
    result = analysis_result()
    result = AnalysisResult(
        **{
            **result.__dict__,
            "messages": ({"role": "user", "content": "hello"},),
            "token_groups": ("User 1",),
        }
    )

    html = result._repr_html_()

    assert "User 1" in html
    assert "&quot;tokens&quot;:[]" in html
    assert "&quot;token_groups&quot;:[{&quot;title&quot;:&quot;User 1&quot;" in html
