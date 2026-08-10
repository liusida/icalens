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
