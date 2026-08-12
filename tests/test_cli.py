from __future__ import annotations

import pytest

from icalens import cli


def test_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--help"])
    output = capsys.readouterr().out
    assert "fit text" in output
    assert "publish" in output


def test_fit_help(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["fit", "--help"])
    output = capsys.readouterr().out
    assert "icalens fit {text,chat}" in output


def test_dispatches_text_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.cli import fit_text

    received: list[str] = []
    monkeypatch.setattr(fit_text, "main", lambda args: received.extend(args))

    cli.main(["fit", "text", "--layers", "6"])

    assert received == ["--layers", "6"]


def test_dispatches_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.cli import publish

    received: list[str] = []
    monkeypatch.setattr(publish, "main", lambda args: received.extend(args))

    cli.main(["publish", "owner/lens", "--lens", "artifact"])

    assert received == ["owner/lens", "--lens", "artifact"]


def test_dispatches_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from icalens.cli import profile

    received: list[str] = []
    monkeypatch.setattr(profile, "main", lambda args: received.extend(args))

    cli.main(["profile", "--lens", "artifact", "--layer", "6"])

    assert received == ["--lens", "artifact", "--layer", "6"]


def test_unknown_command_is_clear() -> None:
    with pytest.raises(SystemExit, match="unknown command"):
        cli.main(["unknown"])
