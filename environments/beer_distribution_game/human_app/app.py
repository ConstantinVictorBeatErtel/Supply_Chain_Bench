"""Gradio UI for anonymous human baselines (Tier 5 Y wholesaler)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

# Allow `python human_app/app.py` from the environment package root.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import gradio as gr

from human_app.session import FIXED_ROLE, FIXED_TIER, HumanSession, PriorExperience
from human_app.session_log import SessionLogger
from human_app.ui_obs import format_observation_html, format_summary_html

_THEME_CSS = (_PKG_ROOT / "human_app" / "theme.css").read_text(encoding="utf-8")

DATA_NOTICE = (
    "Anonymous gameplay data is collected and published openly for research. "
    "Records include a session UUID, task settings, orders, costs, and a "
    "yes/no/unsure experience answer — never names, emails, IP addresses, or free text."
)

_LOGGER = SessionLogger()


def _empty_state() -> dict[str, Any]:
    return {"session": None}


def _idle_station_html() -> str:
    return """
<div class="beer-panel">
  <h2>Your station</h2>
  <p style="margin:0;color:var(--muted);line-height:1.45">
    Answer the question below, then start week 1. You only see your own local
    observation — the same fog-of-war fields an evaluated model receives.
  </p>
</div>
""".strip()


def _obs_html(session: HumanSession | None) -> str:
    if session is None or session.observation is None:
        return _idle_station_html()
    return format_observation_html(session.observation)


def _summary_html(session: HumanSession | None) -> str:
    if session is None or session.status == "in_progress":
        return ""
    return format_summary_html(session.end_summary())


def start_game(experience: str, state: dict[str, Any]):
    if experience not in ("yes", "no", "unsure"):
        raise gr.Error("Please answer whether you have played the beer game before.")
    prior: PriorExperience = experience  # type: ignore[assignment]
    old = state.get("session")
    if isinstance(old, HumanSession) and old.status == "in_progress":
        old.abandon()
        _LOGGER.log_session(old)
    session = HumanSession(prior_beer_game_experience=prior)
    state = {"session": session}
    return (
        state,
        _obs_html(session),
        "",
        gr.update(interactive=True, value=4),
        gr.update(interactive=True),
        gr.update(interactive=True),
        (
            f'You are <strong>Wholesaler</strong> · session '
            f'<strong>{session.session_uuid[:8]}</strong> · week '
            f'<strong>1/{session.episode.spec.horizon}</strong>'
        ),
    )


def place_order(quantity: float | int, state: dict[str, Any]):
    session = state.get("session")
    if not isinstance(session, HumanSession) or session.status != "in_progress":
        raise gr.Error("Start a game before placing an order.")
    try:
        qty = int(quantity)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Order must be an integer from 0 through 128.") from exc
    if qty != quantity:
        raise gr.Error("Order must be an integer from 0 through 128.")
    try:
        session.place_order(qty)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    if session.status == "completed":
        _LOGGER.log_session(session)
        reward = session.end_summary()["episode_reward"]
        return (
            state,
            _idle_station_html(),
            _summary_html(session),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            f"Completed · episode reward <strong>{reward}</strong>",
        )
    week = session.observation["week"]
    horizon = session.observation["horizon"]
    return (
        state,
        _obs_html(session),
        "",
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        (
            f'You are <strong>Wholesaler</strong> · week '
            f'<strong>{week}/{horizon}</strong> · last order <strong>{qty}</strong>'
        ),
    )


def abandon_game(state: dict[str, Any]):
    session = state.get("session")
    if isinstance(session, HumanSession) and session.status == "in_progress":
        session.abandon()
        _LOGGER.log_session(session)
    summary = _summary_html(session if isinstance(session, HumanSession) else None)
    return (
        state,
        _idle_station_html(),
        summary,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        "Session abandoned" if summary else "No active session",
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Beer Distribution Game") as demo:
        state = gr.State(_empty_state())

        # Embed theme in-page so Gradio 5/6 and Spaces all pick it up.
        gr.HTML(
            f"""
<style>{_THEME_CSS}</style>
<div class="beer-hero">
  <p class="beer-kicker">Human baseline · Tier {FIXED_TIER} Y topology</p>
  <h1 class="beer-brand">Beer Distribution Game</h1>
  <p class="beer-tagline">
    Play as the <strong style="color:var(--accent)">Wholesaler</strong> in the
    Y-topology chain for 36 weeks. Retailers and upstream seats are scripted —
    the same seeded task used for LLM evaluation and training. Orders are
    integers from 0–128.
  </p>
  <p class="beer-notice"><strong style="color:var(--accent)">Data notice.</strong> {DATA_NOTICE}</p>
</div>
"""
        )

        with gr.Row():
            experience = gr.Radio(
                choices=["yes", "no", "unsure"],
                label="Have you played the beer game before?",
                value=None,
            )
        with gr.Row():
            start_btn = gr.Button("Start week 1", variant="primary")

        status = gr.HTML(
            '<p class="beer-status">Answer the experience question, then start week 1.</p>'
        )
        observation = gr.HTML(_idle_station_html())

        with gr.Row():
            order = gr.Number(
                label="Place this week’s order",
                value=4,
                precision=0,
                minimum=0,
                maximum=128,
                interactive=False,
            )
        with gr.Row():
            order_btn = gr.Button("Commit order", variant="primary", interactive=False)
            abandon_btn = gr.Button("Abandon session", interactive=False)

        summary = gr.HTML()

        start_btn.click(
            fn=start_game,
            inputs=[experience, state],
            outputs=[
                state,
                observation,
                summary,
                order,
                order_btn,
                abandon_btn,
                status,
            ],
        )
        order_btn.click(
            fn=place_order,
            inputs=[order, state],
            outputs=[
                state,
                observation,
                summary,
                order,
                order_btn,
                abandon_btn,
                status,
            ],
        )
        abandon_btn.click(
            fn=abandon_game,
            inputs=[state],
            outputs=[
                state,
                observation,
                summary,
                order,
                order_btn,
                abandon_btn,
                status,
            ],
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
