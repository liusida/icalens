"""Write self-contained interactive ICA Lens HTML reports."""

# ruff: noqa: E501 -- Long lines keep the embedded HTML/JavaScript template readable.

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def analysis_html(
    result: Any,
    *,
    metric: str = "score",
    top_k: int = 3,
    title: str = "ICA Lens Explorer",
) -> str:
    """Return a self-contained HTML document for an analysis result."""
    payload = _analysis_payload(result, metric=metric, top_k=top_k, title=title, compact=False)
    return _document(payload)


def analysis_iframe(
    result: Any,
    *,
    metric: str = "score",
    top_k: int = 3,
    title: str = "ICA Lens Explorer",
    height: int = 720,
) -> str:
    """Return an isolated interactive notebook representation."""
    if height <= 0:
        raise ValueError("height must be positive")
    payload = _analysis_payload(result, metric=metric, top_k=top_k, title=title, compact=True)
    document = _document(payload)
    source = html.escape(document, quote=True)
    return (
        '<iframe title="ICA Lens Explorer" sandbox="allow-scripts allow-same-origin" '
        f'srcdoc="{source}" style="width:100%;height:{height}px;border:0;overflow:hidden;" '
        'scrolling="no" '
        'loading="lazy"></iframe>'
    )


def write_analysis_html(
    result: Any,
    output_file: str | Path,
    *,
    metric: str = "score",
    top_k: int = 3,
    title: str = "ICA Lens Explorer",
) -> Path:
    """Write an interactive report from an :class:`AnalysisResult`."""
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        analysis_html(result, metric=metric, top_k=top_k, title=title),
        encoding="utf-8",
    )
    return destination


def _analysis_payload(
    result: Any,
    *,
    metric: str,
    top_k: int,
    title: str,
    compact: bool,
) -> str:
    """Serialize the display portion of an analysis result."""
    if metric not in {"score", "energy"}:
        raise ValueError("metric must be 'score' or 'energy'")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scores = result.scores.detach().float().cpu()
    energy = result.energy.detach().float().cpu()
    component_count = int(scores.shape[-1])
    tokens = [
        {
            "position": int(result.positions[row]),
            "token": result.tokens[row],
            "token_text": result.token_texts[row],
            "token_label": result.token_labels[row],
            "token_tooltip": result.token_tooltips[row],
            "metrics": {
                "score": scores[row].tolist(),
                "energy": energy[row].tolist(),
            },
        }
        for row in range(len(result.tokens))
    ]
    token_groups: list[dict[str, Any]] = []
    if result.token_groups:
        for title in dict.fromkeys(result.token_groups):
            grouped = [
                token
                for token, group in zip(tokens, result.token_groups, strict=True)
                if group == title
            ]
            if grouped:
                token_groups.append({"title": title, "open": True, "tokens": grouped})
        tokens = []
    return _payload(
        title=title,
        model=result.model,
        layer=result.layer,
        input_text=result.input_text,
        token_scope=result.token_scope,
        metric=metric,
        top_k=min(top_k, component_count),
        component_count=component_count,
        dynamic_metrics=True,
        compact=compact,
        result_group_title="Tokens",
        messages=list(result.messages),
        tokens=tokens,
        token_groups=token_groups,
        component_profiles=result.component_profiles or {},
    )


def write_explorer_html(
    output_file: str | Path,
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
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(
        title=title,
        model=model,
        layer=layer,
        input_text=input_text,
        token_scope=token_scope,
        metric=metric,
        top_k=max((len(token.get("top", [])) for token in tokens), default=0),
        component_count=0,
        dynamic_metrics=False,
        compact=False,
        result_group_title=result_group_title,
        messages=messages or [],
        tokens=tokens,
        token_groups=token_groups or [],
        component_profiles={},
    )
    destination.write_text(_document(payload), encoding="utf-8")
    return destination


def _payload(**values: Any) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


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
        0 10px 30px rgb(20 25 34/.06); --card-width: 100px;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 3; padding: 12px 18px; background: var(--panel);
      border-bottom: 1px solid var(--border); box-shadow: var(--shadow); }}
    header h1 {{ margin: 0; font-size: 19px; }}
    main {{ max-width: 1680px; margin: 0 auto; padding: 18px; }}
    body.compact header {{ display: none; }}
    body.compact {{ overflow: hidden; }}
    body.compact main {{ padding: 6px; }}
    body.compact .panel {{ margin-bottom: 8px; box-shadow: none; }}
    .panel {{ margin-bottom: 14px; padding: 12px; background: var(--panel);
      border: 1px solid var(--border); border-radius: 8px; box-shadow: var(--shadow); }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 7px; color: var(--muted); }}
    .analysis-context > summary {{ color: var(--muted); font-size: 12px; }}
    .analysis-context[open] > summary {{ margin-bottom: 9px; }}
    .chip {{ padding: 3px 7px; border: 1px solid var(--border); border-radius: 999px;
      background: #f8fafc; }}
    .chip-label {{ color: #445066; font-weight: 750; }}
    .context-label {{ margin-top: 9px; color: #445066; font-size: 11px;
      font-weight: 750; }}
    .input {{ margin: 4px 0 0; padding: 1px 0 1px 10px; overflow-wrap: anywhere;
      white-space: pre-wrap; border-left: 2px solid #c5cfdd; color: #273244;
      font: 500 13px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      letter-spacing: .01em; }}
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
    input, select {{ width: 76px; padding: 5px 7px; border: 1px solid var(--border);
      border-radius: 6px; background: #fff; }}
    button {{ padding: 5px 9px; border: 1px solid var(--border); border-radius: 6px;
      background: #fff; cursor: pointer; font: inherit; font-weight: 700; }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    .selection {{ margin-left: auto; color: var(--muted); font-size: 12px; font-weight: 700; }}
    .results {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--card-width), 1fr));
      gap: 10px; }}
    .token-card {{ min-width: 0; padding: 6px; text-align: center; background: #fff;
      border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 1px 2px rgb(20 25 34/.04); }}
    .token-text {{ min-height: 25px; margin: 1px 0 6px; overflow: hidden;
      white-space: nowrap; text-overflow: ellipsis; font-weight: 850; }}
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
    <details class="panel analysis-context" id="analysisContext">
      <summary>Analysis details</summary>
      <div class="meta">
        <span class="chip" id="model"></span><span class="chip" id="layer"></span>
        <span class="chip" id="scope"></span><span class="chip" id="metric"></span>
        <span class="chip" id="turns" hidden></span><span class="chip" id="count"></span>
      </div>
      <div class="context-label" id="inputLabel">Analyzed text</div>
      <blockquote class="input" id="input"></blockquote>
      <div class="conversation" id="conversation" hidden></div>
    </details>
    <section class="panel controls">
      <label id="metricControl">Metric <select id="metricSelect"><option value="score">Score</option><option value="energy">Energy</option></select></label>
      <label id="topKControl">Top K <input id="topK" type="number" min="1" step="1" value="3"></label>
      <label>Card width <input id="cardWidth" type="number" min="60" max="360" step="20" value="100"></label>
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
    const state = {{ selected: null, metric: data.metric, topK: Number(data.top_k || 3) }};
    const results = document.getElementById("results");
    const cutoff = document.getElementById("cutoff");

    document.body.classList.toggle("compact", Boolean(data.compact));
    document.getElementById("analysisContext").open = !data.compact;
    if (data.compact) {{
      document.querySelector("main").append(document.getElementById("analysisContext"));
    }}
    document.getElementById("title").textContent = data.title;
    const modelChip = document.getElementById("model");
    const revisionAt = data.model.lastIndexOf("@");
    const shortModel = revisionAt < 0
      ? data.model
      : `${{data.model.slice(0, revisionAt)}}@${{data.model.slice(revisionAt + 1, revisionAt + 9)}}`;
    modelChip.innerHTML = `<span class="chip-label">Model:</span> ${{esc(shortModel)}}`;
    modelChip.title = data.model;
    document.getElementById("layer").innerHTML = `<span class="chip-label">Layer:</span> ${{data.layer}}`;
    const scopeLabels = {{
      "all text tokens": "entire input",
      "all": "entire conversation",
      "content": "message content",
      "user": "user messages",
      "assistant": "assistant messages"
    }};
    const scopeLabel = scopeLabels[data.token_scope] || data.token_scope;
    document.getElementById("scope").innerHTML = `<span class="chip-label">Token scope:</span> ${{esc(scopeLabel)}}`;
    document.getElementById("metric").innerHTML = `<span class="chip-label">Metric:</span> ${{esc(state.metric)}}`;
    const metricSelect = document.getElementById("metricSelect");
    const topK = document.getElementById("topK");
    metricSelect.value = state.metric;
    topK.value = state.topK;
    topK.max = Math.max(1, Number(data.component_count || state.topK));
    document.getElementById("metricControl").hidden = !data.dynamic_metrics;
    document.getElementById("topKControl").hidden = !data.dynamic_metrics;
    const totalTokens = data.tokens.length + data.token_groups.reduce(
      (total, group) => total + group.tokens.length, 0
    );
    document.getElementById("count").innerHTML = `<span class="chip-label">Tokens:</span> ${{totalTokens}}`;
    document.getElementById("input").textContent = data.input_text;
    if (data.messages.length) {{
      const input = document.getElementById("input");
      document.getElementById("inputLabel").textContent = "Analyzed conversation";
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
      turns.innerHTML = `<span class="chip-label">User turns:</span> ${{roleCounts.user}}`;
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

    function tokenItems(token) {{
      if (!token.metrics) return token.top;
      const values = token.metrics[state.metric];
      return values.map((score, component) => ({{ component, score: Number(score) }}))
        .sort((left, right) => state.metric === "score"
          ? Math.abs(right.score) - Math.abs(left.score)
          : right.score - left.score)
        .slice(0, state.topK);
    }}

    function tokenCard(token, threshold) {{
      const tokenText = String(token.token_label || token.token_text || token.token).replace(/\\r\\n|\\r|\\n/g, "↵");
      const tokenTooltip = String(token.token_tooltip || token.token);
      const items = tokenItems(token);
      const peak = Math.max(0, ...items.map(item => Math.abs(Number(item.score))));
      const badges = items.map(item => {{
        const component = Number(item.component);
        const score = Number(item.score);
        const ratio = peak ? Math.abs(score) / peak : 0;
        const selected = state.selected === component;
        const bar = selected ? color(component) : "#e7ebf1";
        const display = state.metric === "energy"
          ? `${{(score * 100).toFixed(2)}}%`
          : `${{score >= 0 ? "+" : ""}}${{score.toFixed(3)}}`;
        const tooltip = componentTooltip(component);
        return `<div class="score-row"><button class="badge ${{ratio < threshold ? "weak" : ""}} ${{selected ? "selected" : ""}}" data-component="${{component}}" title="${{esc(tooltip)}}" style="--width:${{(ratio*100).toFixed(1)}}%;--bar:${{bar}};--color:${{color(component)}}"><span class="component">C${{component}}</span><span class="score">${{display}}</span></button></div>`;
      }}).join("");
      return `<article class="token-card"><div class="token-text" title="${{esc(tokenTooltip)}}">${{esc(tokenText)}}</div>${{badges}}</article>`;
    }}

    function componentTooltip(component) {{
      const profile = data.component_profiles[String(component)] || data.component_profiles[component];
      if (!profile) return `C${{component}} · no component profile available`;
      const lines = [`C${{component}} · dominant ${{profile.dominant_sign}}`];
      lines.push("Top occurrences:");
      profile.occurrences.slice(0, 3).forEach((item, index) => {{
        const text = String(item.text || "").replace(/\r\n|\r|\n/g, "↵");
        const context = String(item.context || "").replace(/\r\n|\r|\n/g, "↵");
        lines.push(`${{index + 1}}. ${{JSON.stringify(text)}} · ${{(Number(item.energy) * 100).toFixed(1)}}% · ${{context}}`);
      }});
      lines.push("Top logit-lens tokens:");
      profile.logit_tokens.slice(0, 3).forEach((item, index) => {{
        lines.push(`${{index + 1}}. ${{JSON.stringify(item.text)}} · ${{Number(item.logit) >= 0 ? "+" : ""}}${{Number(item.logit).toFixed(2)}}`);
      }});
      return lines.join("\n");
    }}

    function allTokens() {{
      return [...data.tokens, ...data.token_groups.flatMap(group => group.tokens)];
    }}

    function updateSelection() {{
      const selection = document.getElementById("selection");
      if (state.selected === null) {{
        selection.textContent = "Click a component to highlight it.";
        return;
      }}
      const tokens = allTokens();
      const observations = tokens.map(token => {{
        const raw = token.metrics ? token.metrics[state.metric][state.selected] : null;
        const listed = tokenItems(token).find(item => Number(item.component) === state.selected);
        return {{ token, value: raw === null ? Number(listed?.score) : Number(raw), visible: Boolean(listed) }};
      }}).filter(item => Number.isFinite(item.value));
      const strongest = observations.reduce((best, item) =>
        best === null || Math.abs(item.value) > Math.abs(best.value) ? item : best, null);
      const visible = observations.filter(item => item.visible).length;
      const value = strongest === null ? "unavailable" : state.metric === "energy"
        ? `${{(strongest.value * 100).toFixed(2)}}%`
        : `${{strongest.value >= 0 ? "+" : ""}}${{strongest.value.toFixed(3)}}`;
      const token = strongest === null ? "" : ` at “${{strongest.token.token_text || strongest.token.token}}” (${{value}})`;
      selection.textContent = `C${{state.selected}} · strongest${{token}} · visible in ${{visible}}/${{tokens.length}} tokens`;
    }}

    function resizeFrame() {{
      if (!data.compact || !window.frameElement) return;
      const main = document.querySelector("main");
      const contentHeight = main ? Math.ceil(main.getBoundingClientRect().bottom) : 0;
      const height = Math.max(contentHeight + 4, 180);
      window.frameElement.style.height = `${{height}}px`;
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
        render();
      }}));
      updateSelection();
      requestAnimationFrame(resizeFrame);
    }}
    document.getElementById("cardWidth").addEventListener("input", event => {{
      const value = Math.max(60, Math.min(360, Number(event.target.value || 100)));
      document.documentElement.style.setProperty("--card-width", `${{value}}px`);
      requestAnimationFrame(resizeFrame);
    }});
    metricSelect.addEventListener("change", event => {{
      state.metric = event.target.value;
      document.getElementById("metric").innerHTML = `<span class="chip-label">Metric:</span> ${{esc(state.metric)}}`;
      render();
    }});
    topK.addEventListener("input", event => {{
      state.topK = Math.max(1, Math.min(Number(data.component_count), Number(event.target.value || 3)));
      render();
    }});
    cutoff.addEventListener("input", render);
    document.getElementById("clear").addEventListener("click", () => {{
      state.selected = null;
      document.getElementById("selection").textContent = "Click a component to highlight it.";
      render();
    }});
    render();
    if (data.compact && window.ResizeObserver) new ResizeObserver(resizeFrame).observe(document.body);
  </script>
</body>
</html>
"""
