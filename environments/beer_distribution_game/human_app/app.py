"""Gradio UI for anonymous human baselines (Tier 5 Y wholesaler)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import gradio as gr

from human_app.order_ui import (
    DEFAULT_ORDER,
    ORDER_MAX,
    ORDER_MIN,
    clamp_order,
    nudge_order,
)
from human_app.session import FIXED_TIER, HumanSession
from human_app.session_log import SessionLogger
from human_app.ui_obs import (
    format_masthead_html,
    format_observation_html,
    format_summary_html,
)

_THEME_CSS = (_PKG_ROOT / "human_app" / "theme.css").read_text(encoding="utf-8")
_BEER_THEME = (
    gr.themes.Base()
    .set(
        body_background_fill="#080b0f",
        background_fill_primary="#080b0f",
        background_fill_secondary="#0d141b",
        block_background_fill="transparent",
        block_border_width="0px",
        block_padding="0px",
        block_radius="0px",
        input_background_fill="#0f1922",
        button_primary_background_fill="#f2a53c",
        button_primary_text_color="#0b0f14",
        button_secondary_background_fill="#0f1922",
        button_secondary_text_color="#c7d3dd",
    )
)

_LOGGER = SessionLogger()


def _empty_state() -> dict[str, Any]:
    return {"session": None}


def _briefing_html() -> str:
    return f"""
{format_masthead_html(week=None, horizon=36)}
<div class="beer-briefing">
  <h2>How to play</h2>
  <p>
    You are the <span style="color:var(--accent)">wholesaler</span> in a
    Tier {FIXED_TIER} Y supply chain. Each week the retailer places an order; you
    ship from inventory and reorder from the distributor. Orders take
    <span style="color:var(--accent)">1</span> week to process and shipments take
    <span style="color:var(--accent)">2</span> weeks to arrive. Holding inventory costs
    <span style="color:var(--accent)">$0.5</span>/unit/week; unfilled orders cost
    <span style="color:var(--role-b)">$1.0</span>/unit/week. Keep total cost low over
    <span style="color:var(--accent)">36</span> weeks.
  </p>
  <div class="scripted">
    <strong>Scripted partners</strong>
    <span>
      Other roles in the chain follow the same scripted policies used for LLM
      evaluation. You see only your own local information.
    </span>
  </div>
</div>
""".strip()


def _idle_station_html() -> str:
    return f"""
{_briefing_html()}
<div class="beer-idle">
  <h2>Ready</h2>
  <p style="margin:0">
    Press <strong>Play game</strong> when you are ready. You only see the same
    information an evaluated model receives.
  </p>
</div>
""".strip()


def _debrief_station_html(session: HumanSession) -> str:
    summary = format_summary_html(session.end_summary())
    return f"""
{format_masthead_html(week=None, horizon=36)}
<div class="beer-idle">
  <h2>Game over</h2>
  <p style="margin:0">See your results below, then play again if you like.</p>
</div>
{summary}
""".strip()


def _obs_html(session: HumanSession | None) -> str:
    if session is None or session.observation is None:
        return _idle_station_html()
    return format_observation_html(session.observation)


def _default_order_from_session(session: HumanSession) -> int:
    obs = session.observation
    if obs is None:
        return DEFAULT_ORDER
    last = obs.get("state", {}).get("last_order_placed")
    try:
        value = int(last)
    except (TypeError, ValueError):
        return DEFAULT_ORDER
    return max(ORDER_MIN, min(ORDER_MAX, value if value > 0 else DEFAULT_ORDER))


def _order_input(*, visible: bool, interactive: bool, value: int) -> Any:
    return gr.update(
        visible=visible,
        interactive=interactive,
        value=max(ORDER_MIN, min(ORDER_MAX, int(value))),
    )


def _btn(visible: bool, *, interactive: bool | None = None) -> Any:
    update: dict[str, Any] = {"visible": visible}
    if interactive is not None:
        update["interactive"] = interactive
    return gr.update(**update)


def _start_btn(visible: bool) -> Any:
    return _btn(visible)


def _playing_controls(default: int) -> tuple[Any, ...]:
    return (
        gr.update(visible=True),
        _order_input(visible=True, interactive=True, value=default),
        _btn(True, interactive=True),
        _btn(True, interactive=True),
        gr.update(visible=True),
        _btn(True, interactive=True),
        _btn(True, interactive=True),
    )


def _hidden_controls(default: int = DEFAULT_ORDER) -> tuple[Any, ...]:
    return (
        gr.update(visible=False),
        _order_input(visible=False, interactive=False, value=default),
        _btn(False, interactive=False),
        _btn(False, interactive=False),
        gr.update(visible=False),
        _btn(False, interactive=False),
        _btn(False, interactive=False),
    )


def start_game(state: dict[str, Any]):
    old = state.get("session")
    if isinstance(old, HumanSession) and old.status == "in_progress":
        old.abandon()
        _LOGGER.log_session(old)
    session = HumanSession(prior_beer_game_experience="unsure")
    state = {"session": session}
    default = _default_order_from_session(session)
    controls = _playing_controls(default)
    return (
        state,
        _obs_html(session),
        *controls,
        _start_btn(False),
        _start_btn(False),
    )


def place_order(quantity: float | int, state: dict[str, Any]):
    session = state.get("session")
    if not isinstance(session, HumanSession) or session.status != "in_progress":
        raise gr.Error("Start a game before placing an order.")
    qty = clamp_order(quantity)
    if qty != int(quantity):
        raise gr.Error("Order must be an integer from 0 through 128.")
    try:
        session.place_order(qty)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    if session.status == "completed":
        _LOGGER.log_session(session)
        controls = _hidden_controls()
        return (
            state,
            _debrief_station_html(session),
            *controls,
            _start_btn(False),
            _start_btn(True),
        )

    default = _default_order_from_session(session)
    controls = _playing_controls(default)
    return (
        state,
        _obs_html(session),
        *controls,
        _start_btn(False),
        _start_btn(False),
    )


def abandon_game(state: dict[str, Any]):
    session = state.get("session")
    if isinstance(session, HumanSession) and session.status == "in_progress":
        session.abandon()
        _LOGGER.log_session(session)
    controls = _hidden_controls()
    if isinstance(session, HumanSession) and session.status == "abandoned":
        return (
            state,
            _debrief_station_html(session),
            *controls,
            _start_btn(False),
            _start_btn(True),
        )
    return (
        state,
        _idle_station_html(),
        *controls,
        _start_btn(True),
        _start_btn(False),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Beer Distribution Game",
        fill_height=True,
        fill_width=True,
    ) as demo:
        state = gr.State(_empty_state())

        observation = gr.HTML(_idle_station_html(), container=False)

        start_btn = gr.Button(
            "Play game",
            variant="primary",
            visible=True,
            elem_classes=["beer-btn-square"],
        )
        restart_btn = gr.Button(
            "Play again",
            variant="primary",
            visible=False,
            elem_classes=["beer-btn-square"],
        )

        with gr.Row(
            elem_classes=["beer-order-row"],
            visible=False,
            equal_height=True,
        ) as order_row:
            dec_btn = gr.Button(
                "−",
                elem_classes=["beer-order-btn"],
                interactive=False,
                scale=0,
                min_width=52,
            )
            order_input = gr.Number(
                value=DEFAULT_ORDER,
                precision=0,
                minimum=ORDER_MIN,
                maximum=ORDER_MAX,
                interactive=False,
                show_label=False,
                container=False,
                scale=0,
                min_width=88,
                elem_classes=["beer-order-input"],
            )
            inc_btn = gr.Button(
                "+",
                elem_classes=["beer-order-btn"],
                interactive=False,
                scale=0,
                min_width=52,
            )

        with gr.Row(elem_classes=["beer-action-row"], visible=False) as action_row:
            order_btn = gr.Button(
                "Submit order",
                variant="primary",
                interactive=False,
                elem_classes=["beer-btn-square"],
            )
            abandon_btn = gr.Button(
                "Quit game",
                interactive=False,
                elem_classes=["beer-btn-square", "beer-btn-secondary"],
            )

        play_outputs = [
            state,
            observation,
            order_row,
            order_input,
            dec_btn,
            inc_btn,
            action_row,
            order_btn,
            abandon_btn,
            start_btn,
            restart_btn,
        ]

        start_btn.click(fn=start_game, inputs=[state], outputs=play_outputs)
        restart_btn.click(fn=start_game, inputs=[state], outputs=play_outputs)
        order_btn.click(fn=place_order, inputs=[order_input, state], outputs=play_outputs)
        abandon_btn.click(fn=abandon_game, inputs=[state], outputs=play_outputs)

        dec_btn.click(
            fn=lambda q: nudge_order(q, -1),
            inputs=[order_input],
            outputs=[order_input],
        )
        inc_btn.click(
            fn=lambda q: nudge_order(q, 1),
            inputs=[order_input],
            outputs=[order_input],
        )
    return demo


demo = build_demo()
demo.css = _THEME_CSS
demo.theme = _BEER_THEME

if __name__ == "__main__":
    demo.launch(
        width="100%",
        footer_links=[],
        css=_THEME_CSS,
        theme=_BEER_THEME,
    )
