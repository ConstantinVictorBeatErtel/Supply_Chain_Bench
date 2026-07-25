"""Render Hub observation dicts for humans without leaking extra state."""

from __future__ import annotations

from html import escape
from typing import Any

from human_app.llm_baseline import (
    llm_model_name,
    llm_weekly_costs,
    lookup_llm_baseline,
)

# Exact top-level keys from BeerGameCore.observation (Hub FOW contract).
OBSERVATION_KEYS = frozenset(
    {
        "episode_id",
        "scenario_id",
        "week",
        "horizon",
        "weeks_remaining",
        "role",
        "topology",
        "observation_mode",
        "state",
        "costs",
        "constraints",
        "recent_history",
    }
)

STATE_KEYS_AGGREGATE = frozenset(
    {
        "inventory_on_hand",
        "backlog",
        "inventory_position",
        "on_order",
        "shipment_received",
        "incoming_demand_or_order",
        "units_filled",
        "last_order_placed",
    }
)

STATE_KEYS_SHIPMENT_NOTICES = STATE_KEYS_AGGREGATE | {"inbound_shipment_pipeline"}

COST_KEYS = frozenset(
    {
        "holding_per_unit",
        "backlog_per_unit",
        "current_inventory_backlog_cost",
        "cumulative_local_cost_through_previous_week",
    }
)

CONSTRAINT_KEYS = frozenset(
    {
        "minimum_order",
        "maximum_order",
        "factory_capacity",
    }
)

_CHAIN_NODES = (
    ("F", "Factory", False),
    ("D", "Distributor", False),
    ("W", "You", True),
    ("R", "Retailer", False),
    ("C", "Customer", False),
)


def observation_field_keys(observation: dict[str, Any]) -> set[str]:
    return set(observation)


def assert_observation_parity(observation: dict[str, Any]) -> None:
    """Raise if the observation contains keys outside the Hub FOW contract."""
    extra = set(observation) - OBSERVATION_KEYS
    if extra:
        raise ValueError(f"observation leaked unexpected keys: {sorted(extra)}")
    mode = observation.get("observation_mode")
    allowed_state = (
        STATE_KEYS_SHIPMENT_NOTICES
        if mode == "shipment_notices"
        else STATE_KEYS_AGGREGATE
    )
    state = observation.get("state", {})
    state_extra = set(state) - allowed_state
    if state_extra:
        raise ValueError(f"state leaked unexpected keys: {sorted(state_extra)}")
    cost_extra = set(observation.get("costs", {})) - COST_KEYS
    if cost_extra:
        raise ValueError(f"costs leaked unexpected keys: {sorted(cost_extra)}")
    constraint_extra = set(observation.get("constraints", {})) - CONSTRAINT_KEYS
    if constraint_extra:
        raise ValueError(
            f"constraints leaked unexpected keys: {sorted(constraint_extra)}"
        )


def _role_label(role: str) -> str:
    return role.replace("_", " ").title()


def format_masthead_html(*, week: int | None = None, horizon: int = 36) -> str:
    """Header with week counter and progress segments."""
    if week is None:
        cycle = (
            '<div class="beer-cycle">'
            '<div class="label">Week</div>'
            '<div class="value">—'
            f'<span class="of">/{horizon}</span></div></div>'
        )
        segments = "".join('<span></span>' for _ in range(horizon))
    else:
        cycle = (
            '<div class="beer-cycle">'
            '<div class="label">Week</div>'
            f'<div class="value">{week}'
            f'<span class="of">/{horizon}</span></div></div>'
        )
        bits: list[str] = []
        for i in range(1, horizon + 1):
            if i < week:
                bits.append('<span class="done"></span>')
            elif i == week:
                bits.append('<span class="now"></span>')
            else:
                bits.append("<span></span>")
        segments = "".join(bits)

    return f"""
<div class="beer-masthead">
  <div class="beer-masthead-row">
    <div>
      <h1 class="beer-brand">Beer Distribution Game</h1>
    </div>
    {cycle}
  </div>
  <div class="beer-progress" aria-hidden="true">{segments}</div>
</div>
""".strip()


def format_chain_html() -> str:
    nodes = []
    for initial, name, you in _CHAIN_NODES:
        cls = "beer-node you" if you else "beer-node"
        nodes.append(
            f'<div class="{cls}">'
            f'<div class="badge">{escape(initial)}</div>'
            f'<div class="name">{escape(name)}</div></div>'
        )
    return f"""
<div class="beer-panel">
  <div class="beer-chain-head"><span>Supply Chain</span><span>Goods ▸  ◂ Orders</span></div>
  <div class="beer-chain">{"".join(nodes)}</div>
</div>
""".strip()


C_YOU = "#f2a53c"
C_MODEL = "#7ec8e8"
C_GOOD = "#46d38a"
C_GRID = "rgba(120, 150, 175, 0.22)"
C_AXIS = "#61717f"


def _polyline_from_series(
    values: list[float],
    *,
    width: float = 480.0,
    height: float = 132.0,
    pad: float = 8.0,
    left: float | None = None,
    right: float | None = None,
    top: float | None = None,
    bottom: float | None = None,
    vmax: float | None = None,
) -> tuple[str, float, float, float]:
    if not values:
        return "", pad, height - pad, 0.0
    plot_left = left if left is not None else pad
    plot_right = right if right is not None else pad
    plot_top = top if top is not None else pad
    plot_bottom = bottom if bottom is not None else pad
    peak = max(max(values), 1.0) if vmax is None else max(vmax, 1.0)
    n = len(values)
    points: list[str] = []
    last_x = plot_left
    last_y = height - plot_bottom
    plot_w = width - plot_left - plot_right
    plot_h = height - plot_top - plot_bottom
    for i, v in enumerate(values):
        x = plot_left if n == 1 else plot_left + plot_w * (i / (n - 1))
        y = height - plot_bottom - plot_h * (float(v) / peak)
        points.append(f"{x:.1f},{y:.1f}")
        last_x, last_y = x, y
    return " ".join(points), last_x, last_y, peak


def _chart_legend_html(items: list[tuple[str, str, bool]]) -> str:
    """Legend entries: label, hex color, dashed."""
    parts: list[str] = []
    for label, color, dashed in items:
        if dashed:
            swatch = (
                f'<span class="beer-legend-dash" style="--legend-color:{color}"></span>'
            )
        else:
            swatch = f'<span class="beer-legend-solid" style="background:{color}"></span>'
        parts.append(
            f'<span class="beer-legend-item">{swatch}'
            f"<span>{escape(label)}</span></span>"
        )
    return f'<div class="beer-chart-legend">{"".join(parts)}</div>'


_chart_counter = 0


def _next_chart_id(prefix: str) -> str:
    global _chart_counter
    _chart_counter += 1
    return f"{prefix}{_chart_counter}"


def _y_tick_values(vmax: float, *, count: int = 4) -> list[float]:
    if vmax <= 0:
        return [0.0]
    if count <= 1:
        return [vmax]
    return [vmax * i / (count - 1) for i in range(count)]


def _x_week_labels(n: int) -> list[tuple[int, str]]:
    if n <= 1:
        return [(0, "W1")]
    anchors = [1, max(2, n // 4), max(3, n // 2), max(4, (3 * n) // 4), n]
    seen: set[int] = set()
    out: list[tuple[int, str]] = []
    for w in anchors:
        idx = w - 1
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            out.append((idx, f"W{w}"))
    return out


def _axis_svg(
    *,
    width: float,
    height: float,
    left: float,
    right: float,
    top: float,
    bottom: float,
    vmax: float,
    n_points: int,
    y_prefix: str = "",
    y_suffix: str = "",
) -> str:
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_base = height - bottom
    lines: list[str] = []
    labels: list[str] = []
    for tick in _y_tick_values(vmax):
        if vmax <= 0:
            y = y_base
        else:
            y = y_base - plot_h * (tick / vmax)
        lines.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{width - right:.1f}" '
            f'y2="{y:.1f}" stroke="{C_GRID}" stroke-width="1"></line>'
        )
        label = f"{y_prefix}{tick:g}{y_suffix}"
        labels.append(
            f'<text x="{left - 8:.1f}" y="{y + 3.5:.1f}" text-anchor="end" '
            f'fill="{C_AXIS}" font-size="10">{escape(label)}</text>'
        )
    lines.append(
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{y_base:.1f}" '
        f'stroke="rgba(120,150,175,0.35)" stroke-width="1"></line>'
    )
    lines.append(
        f'<line x1="{left:.1f}" y1="{y_base:.1f}" x2="{width - right:.1f}" '
        f'y2="{y_base:.1f}" stroke="rgba(120,150,175,0.35)" stroke-width="1"></line>'
    )
    for idx, label in _x_week_labels(n_points):
        x = left if n_points == 1 else left + plot_w * (idx / (n_points - 1))
        labels.append(
            f'<text x="{x:.1f}" y="{height - 14:.1f}" text-anchor="middle" '
            f'fill="{C_AXIS}" font-size="10">{escape(label)}</text>'
        )
    labels.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 2:.1f}" text-anchor="middle" '
        f'fill="{C_AXIS}" font-size="10">Week</text>'
    )
    return "".join(lines) + "".join(labels)


def _line_chart_panel(
    *,
    title: str,
    series: list[tuple[str, list[float], str, bool]],
    y_prefix: str = "",
    y_suffix: str = "",
    readout: str = "",
    area_first: bool = False,
    size: str = "normal",
) -> str:
    """Render a multi-series chart with axes and legend."""
    if not series or not series[0][1]:
        return f"""
<div class="beer-panel beer-chart beer-chart-{size}">
  <div class="beer-chart-head">
    <span class="title">{escape(title)}</span>
    <span class="readout" style="color:var(--muted)">—</span>
  </div>
  <p style="margin:0;color:var(--muted);font-size:0.82rem">No data yet.</p>
</div>
""".strip()

    if size == "debrief":
        width, height = 920.0, 300.0
    elif size == "play":
        width, height = 680.0, 240.0
    else:
        width, height = 560.0, 190.0
    left, right, top, bottom = 52.0, 16.0, 20.0, 36.0
    n = max(len(s[1]) for s in series)
    peak = max(max(max(s[1]), 1.0) for s in series)

    axis = _axis_svg(
        width=width,
        height=height,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        vmax=peak,
        n_points=n,
        y_prefix=y_prefix,
        y_suffix=y_suffix,
    )

    legend_items = [(name, color, dashed) for name, _, color, dashed in series]
    legend = _chart_legend_html(legend_items)

    polylines: list[str] = []
    markers: list[str] = []
    area_svg = ""
    fill_id = _next_chart_id("beerFill")
    for i, (_, values, color, dashed) in enumerate(series):
        line, lx, ly, _ = _polyline_from_series(
            values,
            width=width,
            height=height,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            vmax=peak,
        )
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        polylines.append(
            f'<polyline points="{escape(line)}" fill="none" stroke="{color}" '
            f'stroke-width="2.2"{dash} vector-effect="non-scaling-stroke"></polyline>'
        )
        markers.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" fill="{color}"></circle>')
        if area_first and i == 0 and line:
            area_svg = (
                f'<polygon points="{left:.1f},{height - bottom:.1f} {escape(line)} '
                f'{lx:.1f},{height - bottom:.1f}" fill="url(#{fill_id})" '
                f'class="beer-chart-area"></polygon>'
            )
    defs = (
        f'<defs><linearGradient id="{fill_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{C_YOU}" stop-opacity="0.35"/>'
        f'<stop offset="100%" stop-color="{C_YOU}" stop-opacity="0"/>'
        f"</linearGradient></defs>"
        if area_first
        else ""
    )

    readout_html = (
        f'<span class="readout">{readout}</span>' if readout else ""
    )
    return f"""
<div class="beer-panel beer-chart beer-chart-{size}">
  <div class="beer-chart-head">
    <span class="title">{escape(title)}</span>
    {readout_html}
  </div>
  {legend}
  <svg viewBox="0 0 {width:.0f} {height:.0f}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="{escape(title)}">
    {defs}
    {axis}
    {area_svg}
    {"".join(polylines)}
    {"".join(markers)}
  </svg>
</div>
""".strip()


def _inv_backlog_chart_html(recent_history: list[dict[str, Any]]) -> str:
    invs = [float(row.get("ending_inventory", 0)) for row in recent_history]
    backs = [float(row.get("ending_backlog", 0)) for row in recent_history]
    if not invs and not backs:
        return _line_chart_panel(title="Inventory vs backlog", series=[])

    latest_inv = int(invs[-1]) if invs else 0
    latest_back = int(backs[-1]) if backs else 0
    readout = (
        f'<span style="color:{C_GOOD}">{latest_inv}</span>'
        f'<span style="color:var(--muted);font-size:0.75rem"> / </span>'
        f'<span style="color:{C_MODEL}">{latest_back}</span>'
    )
    return _line_chart_panel(
        title="Inventory vs backlog",
        series=[
            ("Inventory (green)", invs, C_GOOD, False),
            ("Backlog (light blue)", backs, C_MODEL, False),
        ],
        readout=readout,
        size="play",
    )


def _cost_chart_html(recent_history: list[dict[str, Any]]) -> str:
    if not recent_history:
        return _line_chart_panel(title="Cumulative cost (your window)", series=[])

    running = 0.0
    cum: list[float] = []
    for row in recent_history:
        running += float(row.get("local_cost", 0))
        cum.append(running)
    latest = cum[-1]
    return _line_chart_panel(
        title="Cumulative cost (your window)",
        series=[("You (yellow)", cum, C_YOU, False)],
        y_prefix="$",
        readout=f'<span style="color:{C_YOU}">${latest:g}</span>',
        area_first=True,
        size="play",
    )


def format_meters_html(observation: dict[str, Any]) -> str:
    assert_observation_parity(observation)
    costs = observation["costs"]
    week_cost = costs["current_inventory_backlog_cost"]
    total = costs["cumulative_local_cost_through_previous_week"]
    return f"""
<div class="beer-meters" aria-live="polite">
  <div class="beer-meter">
    <span class="label">Week</span>
    <span class="value">{observation["week"]}</span>
    <span class="sub">/ {observation["horizon"]}</span>
  </div>
  <div class="beer-meter">
    <span class="label">Your week cost</span>
    <span class="value">{week_cost:g}</span>
  </div>
  <div class="beer-meter">
    <span class="label">Your total</span>
    <span class="value">{total:g}</span>
  </div>
</div>
""".strip()


def format_observation_html(observation: dict[str, Any]) -> str:
    """Station panel HTML using only Hub FOW observation fields."""
    assert_observation_parity(observation)
    state = observation["state"]
    costs = observation["costs"]
    constraints = observation["constraints"]
    role = _role_label(str(observation["role"]))
    demand_label = (
        "Customer demand"
        if str(observation["role"]).startswith("retailer")
        else "Inbound order"
    )
    backlog = int(state["backlog"])
    back_cls = "beer-meter back warn" if backlog > 0 else "beer-meter back"
    total_cost = costs["cumulative_local_cost_through_previous_week"]
    arriving = int(state["shipment_received"])
    on_order = int(state["on_order"])

    pipeline_row = ""
    if "inbound_shipment_pipeline" in state:
        pipe = escape(str(state["inbound_shipment_pipeline"]))
        pipeline_row = f"""
        <div>
          <dt>Ship pipeline</dt>
          <dd style="font-size:0.95rem">{pipe}</dd>
        </div>
        """

    week = int(observation["week"])
    horizon = int(observation["horizon"])

    return f"""
{format_masthead_html(week=week, horizon=horizon)}
<div class="beer-play">
  <div class="beer-col">
    {format_chain_html()}
    <div class="beer-meters" aria-live="polite">
      <div class="beer-meter inv">
        <span class="label">Inventory</span>
        <span class="value">{int(state["inventory_on_hand"])}</span>
        <span class="sub">units on hand</span>
      </div>
      <div class="{back_cls}">
        <span class="label">Backlog</span>
        <span class="value">{backlog}</span>
        <span class="sub">unfilled</span>
      </div>
      <div class="beer-meter transit">
        <span class="label">In Transit</span>
        <span class="value">{on_order}</span>
        <span class="sub">{arriving} inbound now</span>
      </div>
      <div class="beer-meter cost">
        <span class="label">Total Cost</span>
        <span class="value">${total_cost:g}</span>
        <span class="sub">accrued through prior week</span>
      </div>
    </div>
    <div class="beer-inbound">
      <div class="big">{int(state["incoming_demand_or_order"])}</div>
      <div>
        <div class="title">{escape(demand_label)}</div>
        <div class="hint">Retailer order this week. Fill from stock; shortfall → backlog.</div>
      </div>
    </div>
    <div class="beer-panel">
      <h2>Station · {escape(role)}</h2>
      <dl class="beer-stats">
        <div><dt>Received</dt><dd>{arriving}</dd></div>
        <div><dt>On order</dt><dd>{on_order}</dd></div>
        <div><dt>Last order</dt><dd>{int(state["last_order_placed"])}</dd></div>
        <div><dt>Inventory position</dt><dd>{int(state["inventory_position"])}</dd></div>
        <div><dt>Units filled</dt><dd>{int(state["units_filled"])}</dd></div>
        {pipeline_row}
      </dl>
    </div>
  </div>
    <div class="beer-col beer-col-charts">
    {_cost_chart_html(list(observation["recent_history"]))}
    {_inv_backlog_chart_html(list(observation["recent_history"]))}
    <div class="beer-cost-box">
      <h3>How your cost is calculated</h3>
      <p>Week cost = ({costs["holding_per_unit"]} × inventory) + ({costs["backlog_per_unit"]} × backlog)</p>
    </div>
  </div>
</div>
""".strip()


def format_observation_markdown(observation: dict[str, Any]) -> str:
    assert_observation_parity(observation)
    state = observation["state"]
    return (
        f"Week {observation['week']}/{observation['horizon']} · "
        f"Inventory on hand: {state['inventory_on_hand']}"
    )


def _fmt_cost(value: Any) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.1f}"


def _fmt_reward(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def _verdict(you: float, other: float) -> tuple[str, str]:
    margin = abs(you - other)
    if you < other - 1e-9:
        return "You beat this baseline", f"${margin:,.1f} lower cost"
    if you > other + 1e-9:
        return "Baseline wins", f"${margin:,.1f} lower cost"
    return "Tied", "$0.0 margin"


def _comparison_charts_html(
    *,
    you_actions: list[int],
    you_weekly: list[dict[str, Any]],
    llm_actions: list[int] | None,
    llm_weekly_costs: list[float] | None,
    model_name: str,
) -> str:
    if not you_actions:
        return ""
    you_costs = [float(row.get("local_cost", 0)) for row in you_weekly]
    weeks = min(len(you_actions), len(you_costs))
    if weeks == 0:
        return ""

    ya = [float(v) for v in you_actions[:weeks]]
    yc: list[float] = []
    running = 0.0
    for c in you_costs[:weeks]:
        running += c
        yc.append(running)

    la: list[float] | None = None
    lc: list[float] | None = None
    if llm_actions is not None and llm_weekly_costs is not None:
        n = min(weeks, len(llm_actions), len(llm_weekly_costs))
        la = [float(v) for v in llm_actions[:n]]
        running = 0.0
        lc = []
        for c in llm_weekly_costs[:n]:
            running += float(c)
            lc.append(running)
        weeks = n
        ya = ya[:weeks]
        yc = yc[:weeks]

    note = (
        f"Compared to {escape(model_name)} on this exact seed "
        "(recorded Hub eval orders, replayed here — not a live API call during your game)."
    )
    if la is None:
        note = "No stored LLM trajectory for this seed; only your play is charted."

    order_series: list[tuple[str, list[float], str, bool]] = [
        ("You (yellow)", ya, C_YOU, False),
    ]
    cost_series: list[tuple[str, list[float], str, bool]] = [
        ("You (yellow)", yc, C_YOU, False),
    ]
    if la is not None and lc is not None:
        order_series.append((f"{model_name} (light blue)", la, C_MODEL, True))
        cost_series.append((f"{model_name} (light blue)", lc, C_MODEL, True))

    orders_chart = _line_chart_panel(
        title="Weekly orders",
        series=order_series,
        y_suffix=" u",
        size="debrief",
    )
    cost_chart = _line_chart_panel(
        title="Cumulative cost",
        series=cost_series,
        y_prefix="$",
        area_first=True,
        size="debrief",
    )

    return f"""
<div class="beer-debrief-charts">
  <p class="beer-debrief-note">{note}</p>
  {orders_chart}
  {cost_chart}
</div>
""".strip()


def format_summary_html(summary: dict[str, Any]) -> str:
    if not summary or summary.get("status") == "in_progress":
        return ""
    status = str(summary["status"])
    abandoned = status != "completed"
    title = "Session complete" if not abandoned else "Session abandoned"
    cls = "beer-summary abandoned" if abandoned else "beer-summary"

    rows: list[tuple[str, Any]] = []
    compare_html = ""
    charts_html = ""
    if not abandoned:
        you_cost = float(summary["final_total_cost"])
        base_cost = summary.get("base_stock_cost")
        llm = lookup_llm_baseline(str(summary["split"]), int(summary["seed_index"]))
        llm_cost = None if llm is None else llm["local_total_cost"]
        llm_reward = None if llm is None else llm["reward"]
        your_reward = summary.get("episode_reward")

        vs_base_verdict, vs_base_margin = (
            _verdict(you_cost, float(base_cost)) if base_cost is not None else ("—", "")
        )
        if llm_cost is not None:
            vs_llm_verdict, vs_llm_margin = _verdict(you_cost, float(llm_cost))
        else:
            vs_llm_verdict, vs_llm_margin = ("—", "No LLM baseline for this seed")

        llm_actions = None if llm is None else llm.get("actions")
        llm_costs = llm_weekly_costs(str(summary["split"]), int(summary["seed_index"]))
        charts_html = _comparison_charts_html(
            you_actions=[int(a) for a in summary.get("actions", [])],
            you_weekly=list(summary.get("weekly", [])),
            llm_actions=llm_actions,
            llm_weekly_costs=llm_costs,
            model_name=llm_model_name(),
        )

        compare_html = f"""
<div class="beer-compare-grid">
  <div class="beer-compare-card you">
    <div class="label">You</div>
    <div class="cost">{_fmt_cost(you_cost)}</div>
    <div class="sub">reward {_fmt_reward(your_reward)}</div>
  </div>
  <div class="beer-compare-card llm">
    <div class="label">{escape(llm_model_name())}</div>
    <div class="cost">{_fmt_cost(llm_cost)}</div>
    <div class="sub">reward {_fmt_reward(llm_reward)} · same seed eval</div>
    <div class="verdict">{escape(vs_llm_verdict)} · {escape(vs_llm_margin)}</div>
  </div>
  <div class="beer-compare-card base">
    <div class="label">Adaptive base-stock</div>
    <div class="cost">{_fmt_cost(base_cost)}</div>
    <div class="sub">same-seed scripted baseline</div>
    <div class="verdict">{escape(vs_base_verdict)} · {escape(vs_base_margin)}</div>
  </div>
</div>
"""
    else:
        rows.append(
            (
                "Note",
                "Abandoned before week 36; partial actions were logged anonymously.",
            )
        )

    body = "".join(
        f'<div class="row"><span>{escape(str(k))}</span>'
        f"<span>{escape(str(v))}</span></div>"
        for k, v in rows
    )
    return f"""
<section class="{cls}">
  <div class="eyebrow">Results</div>
  <h2>{escape(title)}</h2>
  {compare_html}
  {charts_html}
  {body}
</section>
""".strip()
