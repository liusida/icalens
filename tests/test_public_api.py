from __future__ import annotations

import icalens


def test_public_api_is_small() -> None:
    assert icalens.__all__ == ["ArtifactError", "ICALens", "ICALensError", "NotFittedError"]
    assert icalens.__version__ == "0.2.0.dev0"
