#!/usr/bin/env python3
"""Render selected MLP-neuron profiles as a self-contained tabbed HTML report."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "runs/main/selected-neuron-profiles-context.json"
DEFAULT_OUTPUT = HERE / "reports/neuron-profiles.html"
GROUP_LABELS = {
    "lowest": "Lowest",
    "above_median_random": "Above-median random",
    "highest": "Highest",
}
GROUP_CLASSES = {
    "lowest": "low",
    "above_median_random": "middle",
    "highest": "high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace an existing report.")
    return parser.parse_args()


def load_profiles(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 9:
        raise ValueError("expected exactly nine selected-neuron profiles")
    expected_groups = ["lowest"] * 3 + ["above_median_random"] * 3 + ["highest"] * 3
    seen: set[int] = set()
    for profile, expected_group in zip(profiles, expected_groups, strict=True):
        neuron = profile.get("neuron")
        occurrences = profile.get("occurrences")
        if (
            not isinstance(neuron, int)
            or neuron in seen
            or profile.get("selection_group") != expected_group
            or profile.get("dominant_sign") not in {"positive", "negative"}
            or not isinstance(occurrences, list)
            or len(occurrences) != 20
        ):
            raise ValueError(f"invalid selected-neuron profile: {profile}")
        seen.add(neuron)
        for occurrence in occurrences:
            context = occurrence.get("context")
            start = occurrence.get("context_target_start")
            end = occurrence.get("context_target_end")
            if (
                not isinstance(context, str)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(context)
            ):
                raise ValueError(f"invalid occurrence span for neuron {neuron}")
    return profiles


def occurrence_html(occurrence: dict[str, Any], rank: int) -> str:
    score = float(occurrence["score"])
    context_tokens = occurrence.get("context_tokens")
    if not isinstance(context_tokens, list) or not context_tokens:
        raise ValueError("report input lacks tokenwise scored contexts; run enrich_contexts.py")
    rendered_tokens = []
    for token in context_tokens:
        token_score = float(token["score"])
        # Fixed linear scale across every neuron and occurrence. Scores below
        # ten remain deliberately pale; unusually large scores become darker.
        alpha = 0.02 + 0.56 * min(abs(token_score) / 25.0, 1.0)
        rgb = "61,95,153" if token_score >= 0 else "180,95,77"
        target = " target" if bool(token.get("target")) else ""
        rendered_tokens.append(
            f'<span class="scored-token{target}" style="background:rgba({rgb},{alpha:.3f})" '
            f'title="position {int(token["position"])} · score {token_score:+.4f}">'
            f'{html.escape(str(token["text"]))}</span>'
        )
    return f"""
      <li class="occurrence">
        <div class="occurrence-head">
          <span class="rank">#{rank}</span>
          <span class="score">score {score:+.4f}</span>
          <span class="location">doc {int(occurrence['source_index'])} · position {int(occurrence['position'])}</span>
        </div>
        <pre>{''.join(rendered_tokens)}</pre>
        <div class="token-meta">token <code>{html.escape(str(occurrence['token']))}</code> · id {int(occurrence['token_id'])}</div>
      </li>"""


def panel_html(profile: dict[str, Any], index: int) -> str:
    neuron = int(profile["neuron"])
    group = str(profile["selection_group"])
    group_label = GROUP_LABELS[group]
    occurrences = "".join(
        occurrence_html(occurrence, rank)
        for rank, occurrence in enumerate(profile["occurrences"], start=1)
    )
    active = " active" if index == 0 else ""
    hidden = "" if index == 0 else " hidden"
    return f"""
    <section id="panel-{index}" class="tab-panel{active}" role="tabpanel" aria-labelledby="tab-{index}"{hidden}>
      <header class="profile-header">
        <div>
          <div class="eyebrow {GROUP_CLASSES[group]}">{html.escape(group_label)}</div>
          <h2>Layer 16 · Neuron {neuron}</h2>
          <p>{html.escape(str(profile['down_projection_direction']))}</p>
        </div>
        <div class="tail-badge {html.escape(str(profile['dominant_sign']))}">{html.escape(str(profile['dominant_sign']).title())} tail</div>
      </header>
      <div class="metrics">
        <div><span>Excess kurtosis</span><strong>{float(profile['excess_kurtosis']):.3f}</strong></div>
        <div><span>Skewness</span><strong>{float(profile['skewness']):+.3f}</strong></div>
        <div><span>Variance</span><strong>{float(profile['variance']):.4g}</strong></div>
        <div><span>Mean</span><strong>{float(profile['mean']):+.4g}</strong></div>
      </div>
      <div class="occurrence-title">
        <h3>Top 20 dominant-tail occurrences</h3>
        <div class="color-key"><span class="negative-swatch"></span> negative <span class="positive-swatch"></span> positive · fixed linear intensity · target is underlined</div>
      </div>
      <ol class="occurrences">{occurrences}</ol>
    </section>"""


def render(profiles: list[dict[str, Any]]) -> str:
    tabs = []
    panels = []
    for index, profile in enumerate(profiles):
        group = str(profile["selection_group"])
        active = " active" if index == 0 else ""
        selected = "true" if index == 0 else "false"
        tabs.append(
            f'<button id="tab-{index}" class="tab {GROUP_CLASSES[group]}{active}" '
            f'role="tab" aria-selected="{selected}" aria-controls="panel-{index}" '
            f'data-panel="panel-{index}"><span>{html.escape(GROUP_LABELS[group])}</span>'
            f'<strong>N{int(profile["neuron"])}</strong></button>'
        )
        panels.append(panel_html(profile, index))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen layer-16 MLP neuron profiles</title>
<style>
  :root {{ color-scheme: light; --blue:#3d5f99; --ink:#18202c; --muted:#667085; --line:#d8dee8; --bg:#f5f7fa; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
  main {{ max-width:1180px; margin:0 auto; padding:30px 22px 60px; }}
  h1 {{ margin:0 0 5px; font-size:26px; letter-spacing:-.02em; }}
  .subtitle {{ margin:0 0 24px; color:var(--muted); }}
  .tabs {{ display:grid; grid-template-columns:repeat(9,minmax(92px,1fr)); gap:7px; margin-bottom:12px; }}
  .tab {{ border:1px solid var(--line); border-top:4px solid #9aa4b2; border-radius:8px 8px 4px 4px; background:white; padding:8px 7px; cursor:pointer; color:var(--ink); }}
  .tab span {{ display:block; min-height:30px; font-size:10px; line-height:1.15; color:var(--muted); }}
  .tab strong {{ font-size:14px; }}
  .tab.low {{ border-top-color:#7a8da9; }} .tab.middle {{ border-top-color:#c28a45; }} .tab.high {{ border-top-color:#b45f4d; }}
  .tab.active {{ outline:2px solid var(--blue); outline-offset:1px; background:#f8fbff; }}
  .tab-panel {{ background:white; border:1px solid var(--line); border-radius:10px; padding:24px; box-shadow:0 5px 20px rgba(21,34,50,.05); }}
  .profile-header {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-start; border-bottom:1px solid var(--line); padding-bottom:16px; }}
  .profile-header h2 {{ margin:2px 0 3px; font-size:24px; }} .profile-header p {{ margin:0; color:var(--muted); }}
  .eyebrow {{ font-size:12px; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }}
  .eyebrow.low {{ color:#617591; }} .eyebrow.middle {{ color:#9a6427; }} .eyebrow.high {{ color:#9d4637; }}
  .tail-badge {{ flex:none; border-radius:999px; padding:6px 11px; font-size:12px; font-weight:700; }}
  .tail-badge.positive {{ color:#176b4d; background:#e7f5ef; }} .tail-badge.negative {{ color:#963e34; background:#f9e9e7; }}
  .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:18px 0 22px; }}
  .metrics div {{ background:#f7f9fc; border:1px solid #e5e9f0; border-radius:8px; padding:11px 13px; }}
  .metrics span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
  .metrics strong {{ display:block; margin-top:2px; font:700 19px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace; }}
  h3 {{ margin:0 0 12px; font-size:16px; }}
  .occurrence-title {{ display:flex; justify-content:space-between; align-items:baseline; gap:16px; }}
  .color-key {{ color:var(--muted); font-size:10px; white-space:nowrap; }}
  .color-key span {{ display:inline-block; width:11px; height:11px; border-radius:2px; vertical-align:-2px; margin:0 3px 0 7px; }}
  .negative-swatch {{ background:rgba(180,95,77,.58); }} .positive-swatch {{ background:rgba(61,95,153,.58); }}
  .occurrences {{ list-style:none; padding:0; margin:0; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
  .occurrence {{ min-width:0; border:1px solid #e1e6ee; border-radius:8px; padding:10px 12px; background:#fcfdff; }}
  .occurrence-head {{ display:flex; align-items:center; gap:10px; margin-bottom:7px; color:var(--muted); font-size:11px; }}
  .rank {{ color:var(--ink); font-weight:800; }} .score {{ color:var(--blue); font:700 11px ui-monospace, monospace; }} .location {{ margin-left:auto; }}
  pre {{ margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font:13px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .scored-token {{ border-radius:2px; }}
  .scored-token.target {{ text-decoration:underline 1.5px; text-underline-offset:3px; }}
  .token-meta {{ margin-top:7px; color:var(--muted); font-size:10px; }} code {{ color:#344054; }}
  [hidden] {{ display:none !important; }}
  @media (max-width:900px) {{ .tabs {{ grid-template-columns:repeat(3,1fr); }} .occurrences {{ grid-template-columns:1fr; }} .occurrence-title {{ display:block; }} .color-key {{ margin:-5px 0 12px; white-space:normal; }} }}
  @media (max-width:560px) {{ main {{ padding:18px 10px 40px; }} .metrics {{ grid-template-columns:repeat(2,1fr); }} .profile-header {{ display:block; }} .tail-badge {{ display:inline-block; margin-top:10px; }} }}
</style>
</head>
<body>
<main>
  <h1>Qwen 3.5 9B Base · layer 16 MLP neurons</h1>
  <p class="subtitle">Nine neurons selected after measuring all 12,288 neurons on one million fitting-token positions. Occurrences come from the fixed 100,000-token profiling subset.</p>
  <nav class="tabs" role="tablist" aria-label="Selected neurons">{''.join(tabs)}</nav>
  {''.join(panels)}
</main>
<script>
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const panels = [...document.querySelectorAll('[role="tabpanel"]')];
  function activate(tab) {{
    tabs.forEach(item => {{ const on = item === tab; item.classList.toggle('active', on); item.setAttribute('aria-selected', String(on)); }});
    panels.forEach(panel => {{ const on = panel.id === tab.dataset.panel; panel.classList.toggle('active', on); panel.hidden = !on; }});
    history.replaceState(null, '', '#' + tab.dataset.panel);
  }}
  tabs.forEach((tab, index) => {{
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', event => {{
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();
      let target = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      tabs[target].focus(); activate(tabs[target]);
    }});
  }});
  const requested = document.querySelector(location.hash);
  if (requested?.getAttribute('role') === 'tabpanel') {{ const tab = tabs.find(item => item.dataset.panel === requested.id); if (tab) activate(tab); }}
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"report already exists; pass --force to replace it: {output}")
    profiles = load_profiles(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(render(profiles), encoding="utf-8")
    if "role=\"tablist\"" not in temporary.read_text(encoding="utf-8"):
        raise ValueError("temporary report failed validation")
    os.replace(temporary, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
