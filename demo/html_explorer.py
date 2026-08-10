"""Write a self-contained, v5-style token/component explorer."""

# ruff: noqa: E501 -- Long lines keep the embedded HTML/JavaScript template readable.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_explorer_html(
    output_file: Path,
    *,
    title: str,
    model: str,
    layer: int,
    input_text: str,
    token_scope: str,
    tokens: list[dict[str, Any]],
    token_groups: list[dict[str, Any]] | None = None,
    metric: str = "score",
    result_group_title: str = "Tokens",
    messages: list[dict[str, str]] | None = None,
) -> Path:
    """Write an interactive standalone HTML report and return its absolute path."""
    if metric not in {"score", "energy"}:
        raise ValueError("metric must be 'score' or 'energy'")
    destination = output_file.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "title": title,
            "model": model,
            "layer": layer,
            "input_text": input_text,
            "token_scope": token_scope,
            "metric": metric,
            "result_group_title": result_group_title,
            "messages": messages or [],
            "tokens": tokens,
            "token_groups": token_groups or [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    destination.write_text(_document(payload), encoding="utf-8")
    return destination


def _document(payload: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ICA Lens Explorer</title>
  <style>
    :root {{
      --bg: #f6f7f9; --panel: #fff; --text: #151922; --muted: #647084;
      --border: #cbd3df; --accent: #1f6feb; --shadow: 0 1px 2px rgb(20 25 34/.08),
        0 10px 30px rgb(20 25 34/.06); --card-width: 140px;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 3; padding: 12px 18px; background: var(--panel);
      border-bottom: 1px solid var(--border); box-shadow: var(--shadow); }}
    header h1 {{ margin: 0; font-size: 19px; }}
    main {{ max-width: 1680px; margin: 0 auto; padding: 18px; }}
    .panel {{ margin-bottom: 14px; padding: 12px; background: var(--panel);
      border: 1px solid var(--border); border-radius: 8px; box-shadow: var(--shadow); }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 7px; color: var(--muted); }}
    .chip {{ padding: 3px 7px; border: 1px solid var(--border); border-radius: 999px;
      background: #f8fafc; }}
    .input {{ margin: 10px 0 0; padding: 9px; overflow-wrap: anywhere; white-space: pre-wrap;
      border: 1px solid #dde3ec; border-radius: 6px; background: #f8fafc;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .conversation {{ display: grid; gap: 9px; margin-top: 12px; }}
    .message {{ width: min(88%, 1050px); padding: 9px 11px; border: 1px solid #d8e0eb;
      border-radius: 9px; background: #f8fafc; }}
    .message.user {{ margin-left: auto; background: #eef5ff; border-color: #bfd3f5; }}
    .message.assistant {{ margin-right: auto; background: #f7f8fa; }}
    .message.system {{ width: 100%; background: #fff8e8; border-color: #ead7a4; }}
    .message-role {{ margin-bottom: 3px; color: var(--muted); font-size: 11px;
      font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }}
    .message-content {{ overflow-wrap: anywhere; white-space: pre-wrap; }}
    details > summary {{ cursor: pointer; font-weight: 800; }}
    details .results {{ margin-top: 12px; }}
    .controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }}
    label {{ display: flex; align-items: center; gap: 7px; color: #435066;
      font-size: 12px; font-weight: 700; }}
    input {{ width: 76px; padding: 5px 7px; border: 1px solid var(--border); border-radius: 6px; }}
    button {{ padding: 5px 9px; border: 1px solid var(--border); border-radius: 6px;
      background: #fff; cursor: pointer; font: inherit; font-weight: 700; }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    .selection {{ margin-left: auto; color: var(--muted); font-size: 12px; font-weight: 700; }}
    .results {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--card-width), 1fr));
      gap: 10px; }}
    .token-card {{ min-width: 0; padding: 6px; text-align: center; background: #fff;
      border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 1px 2px rgb(20 25 34/.04); }}
    .token-position {{ color: #94a3b8; font-size: 10px; }}
    .token-text {{ min-height: 25px; margin: 1px 0 6px; overflow-wrap: anywhere;
      white-space: pre-wrap; font-weight: 850; }}
    .score-row {{ position: relative; min-height: 24px; margin-top: 3px; }}
    .badge {{ position: relative; display: flex; width: 100%; min-height: 24px; align-items: center;
      justify-content: space-between; gap: 5px; padding: 3px 5px; overflow: hidden;
      border: 1px solid #d5dce7; border-radius: 5px; color: var(--text); font-size: 10px;
      background: linear-gradient(90deg, var(--bar) 0, var(--bar) var(--width),
        transparent var(--width), transparent 100%); }}
    .badge.weak {{ opacity: .35; }}
    .badge.selected {{ outline: 2px solid var(--color); border-color: var(--color); opacity: 1; }}
    .component {{ font-weight: 850; }}
    .score {{ font-variant-numeric: tabular-nums; }}
    @media (max-width: 700px) {{ main {{ padding: 10px; }} .selection {{ flex-basis: 100%; }} }}
  </style>
</head>
<body>
  <header><h1 id="title">ICA Lens Explorer</h1></header>
  <main>
    <section class="panel">
      <div class="meta">
        <span class="chip" id="model"></span><span class="chip" id="layer"></span>
        <span class="chip" id="scope"></span><span class="chip" id="metric"></span>
        <span class="chip" id="turns" hidden></span><span class="chip" id="count"></span>
      </div>
      <div class="input" id="input"></div>
      <div class="conversation" id="conversation" hidden></div>
    </section>
    <section class="panel controls">
      <label>Card width <input id="cardWidth" type="number" min="100" max="360" value="140"></label>
      <label>Opacity cutoff <input id="cutoff" type="number" min="0" max="1" step="0.05" value="0.5"></label>
      <button id="clear" type="button">Clear selection</button>
      <span class="selection" id="selection">Click a component to highlight it.</span>
    </section>
    <div id="tokenGroups"></div>
    <details class="panel" id="resultGroup" open>
      <summary id="resultSummary"></summary>
      <section class="results" id="results"></section>
    </details>
  </main>
  <script>
    const data = {payload};
    const esc = value => String(value).replace(/[&<>"']/g, char => ({{
      "&":"&amp;", "<":"&lt;", ">":"&gt;", "\\\"":"&quot;", "'":"&#39;"
    }})[char]);
    const color = component => `hsl(${{(Number(component) * 137.508) % 360}} 66% 78%)`;
    const state = {{ selected: null }};
    const results = document.getElementById("results");
    const cutoff = document.getElementById("cutoff");

    document.getElementById("title").textContent = data.title;
    document.getElementById("model").textContent = data.model;
    document.getElementById("layer").textContent = `layer ${{data.layer}}`;
    document.getElementById("scope").textContent = data.token_scope;
    document.getElementById("metric").textContent = data.metric;
    const totalTokens = data.tokens.length + data.token_groups.reduce(
      (total, group) => total + group.tokens.length, 0
    );
    document.getElementById("count").textContent = `${{totalTokens}} tokens`;
    document.getElementById("input").textContent = data.input_text;
    if (data.messages.length) {{
      const input = document.getElementById("input");
      const conversation = document.getElementById("conversation");
      const turns = document.getElementById("turns");
      input.hidden = true;
      conversation.hidden = false;
      const roleCounts = {{ user: 0, assistant: 0, system: 0 }};
      data.messages.forEach(message => {{
        const role = String(message.role);
        roleCounts[role] = (roleCounts[role] || 0) + 1;
        const card = document.createElement("article");
        card.className = `message ${{role}}`;
        const label = document.createElement("div");
        label.className = "message-role";
        label.textContent = role === "system"
          ? "System"
          : `${{role === "user" ? "User" : "Assistant"}} ${{roleCounts[role]}}`;
        const content = document.createElement("div");
        content.className = "message-content";
        content.textContent = message.content;
        card.append(label, content);
        conversation.append(card);
      }});
      turns.hidden = false;
      turns.textContent = `${{roleCounts.user}} turn${{roleCounts.user === 1 ? "" : "s"}}`;
    }}
    document.getElementById("resultSummary").textContent =
      `${{data.result_group_title}} (${{data.tokens.length}} tokens)`;
    document.getElementById("resultGroup").hidden = data.tokens.length === 0;
    const tokenGroups = document.getElementById("tokenGroups");
    const groupResults = [];
    data.token_groups.forEach(group => {{
      const details = document.createElement("details");
      details.className = "panel";
      details.open = Boolean(group.open);
      const summary = document.createElement("summary");
      summary.textContent = `${{group.title}} (${{group.tokens.length}} tokens)`;
      const cards = document.createElement("section");
      cards.className = "results";
      details.append(summary, cards);
      tokenGroups.append(details);
      groupResults.push({{ cards, tokens: group.tokens }});
    }});

    function tokenCard(token, threshold) {{
      const tokenText = String(token.token_text || token.token).replace(/\\r\\n|\\r|\\n/g, "↵");
      const peak = Math.max(0, ...token.top.map(item => Math.abs(Number(item.score))));
      const badges = token.top.map(item => {{
        const component = Number(item.component);
        const score = Number(item.score);
        const ratio = peak ? Math.abs(score) / peak : 0;
        const selected = state.selected === component;
        const bar = selected ? color(component) : "#e7ebf1";
        const display = data.metric === "energy"
          ? `${{(score * 100).toFixed(2)}}%`
          : `${{score >= 0 ? "+" : ""}}${{score.toFixed(3)}}`;
        return `<div class="score-row"><button class="badge ${{ratio < threshold ? "weak" : ""}} ${{selected ? "selected" : ""}}" data-component="${{component}}" style="--width:${{(ratio*100).toFixed(1)}}%;--bar:${{bar}};--color:${{color(component)}}"><span class="component">C${{component}}</span><span class="score">${{display}}</span></button></div>`;
      }}).join("");
      return `<article class="token-card"><div class="token-position">${{token.position}}</div><div class="token-text" title="${{esc(token.token)}}">${{esc(tokenText)}}</div>${{badges}}</article>`;
    }}

    function render() {{
      const threshold = Math.max(0, Math.min(1, Number(cutoff.value || .5)));
      results.innerHTML = data.tokens.map(token => tokenCard(token, threshold)).join("");
      groupResults.forEach(group => {{
        group.cards.innerHTML = group.tokens.map(token => tokenCard(token, threshold)).join("");
      }});
      document.querySelectorAll(".badge").forEach(node => node.addEventListener("click", () => {{
        const component = Number(node.dataset.component);
        state.selected = state.selected === component ? null : component;
        document.getElementById("selection").textContent = state.selected === null
          ? "Click a component to highlight it." : `Selected C${{state.selected}}`;
        render();
      }}));
    }}
    document.getElementById("cardWidth").addEventListener("input", event => {{
      const value = Math.max(100, Math.min(360, Number(event.target.value || 140)));
      document.documentElement.style.setProperty("--card-width", `${{value}}px`);
    }});
    cutoff.addEventListener("input", render);
    document.getElementById("clear").addEventListener("click", () => {{
      state.selected = null;
      document.getElementById("selection").textContent = "Click a component to highlight it.";
      render();
    }});
    render();
  </script>
</body>
</html>
"""
