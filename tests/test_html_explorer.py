from __future__ import annotations

from demo.html_explorer import write_explorer_html


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
    assert 'details.open = Boolean(group.open)' in html
    assert 'group.tokens.map(token => tokenCard(token, threshold))' in html
    assert '"metric":"score"' in html
    assert 'data.metric === "energy"' in html
    assert 'replace(/\\r\\n|\\r|\\n/g, "↵")' in html
    assert "max-width: 1680px" in html
    assert '`${data.result_group_title} (${data.tokens.length} tokens)`' in html
    assert '<details class="panel" id="resultGroup" open>' in html
    assert '"content":"Second turn"' in html
    assert 'turns.textContent = `${roleCounts.user} turn' in html
    assert 'card.className = `message ${role}`' in html
    assert html.index('<section class="panel controls">') < html.index(
        '<div id="tokenGroups">'
    )


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
