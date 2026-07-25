"""Gradio UI for anonymous human baselines (Tier 5 Y wholesaler)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow `python human_app/app.py` from the environment package root.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import gradio as gr

from human_app.session_log import SessionLogger
from human_app.session import FIXED_ROLE, FIXED_TIER, HumanSession, PriorExperience
from human_app.ui_obs import format_observation_markdown

DATA_NOTICE = (
    "**Data collection notice.** Anonymous gameplay data from this app is collected "
    "and published openly for research. Records contain a session UUID, task "
    "settings, orders, costs, and a one-question experience self-report. "
    "No names, emails, IP addresses, or free-text answers are collected."
)

TASK_BLURB = (
    f"You play the **{FIXED_ROLE}** in the Tier {FIXED_TIER} Y-topology beer game "
    "(36 weeks, orders 0–128). Other roles are scripted adaptive base-stock "
    "policies — the same task used for LLM evaluation. You only see your local "
    "observation each week."
)

_LOGGER = SessionLogger()


def _empty_state() -> dict[str, Any]:
    return {"session": None}


def _obs_md(session: HumanSession | None) -> str:
    if session is None or session.observation is None:
        return "_No active observation._"
    return format_observation_markdown(session.observation)


def _summary_md(session: HumanSession | None) -> str:
    if session is None or session.status == "in_progress":
        return ""
    s = session.end_summary()
    lines = [
        f"### Session {s['status']}",
        f"- Weeks played: {s['weeks_played']}",
        f"- Seed: `{s['seed']}` ({s['split']} #{s['seed_index']})",
    ]
    if s["status"] == "completed":
        lines.extend(
            [
                f"- Final total cost: **{s['final_total_cost']}**",
                f"- Same-seed base-stock cost: **{s['base_stock_cost']}**",
                f"- Episode reward: **{s['episode_reward']}**",
            ]
        )
    else:
        lines.append("_Abandoned before the 36-week horizon; partial actions were logged._")
    return "\n".join(lines)


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
        _obs_md(session),
        "",
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        f"Session `{session.session_uuid}` · week 1/{session.episode.spec.horizon}",
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
        return (
            state,
            _obs_md(session),
            _summary_md(session),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            f"Completed · reward={session.end_summary()['episode_reward']}",
        )
    return (
        state,
        _obs_md(session),
        "",
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        f"Week {session.observation['week']}/{session.observation['horizon']} · "
        f"last order {qty}",
    )


def abandon_game(state: dict[str, Any]):
    session = state.get("session")
    if isinstance(session, HumanSession) and session.status == "in_progress":
        session.abandon()
        _LOGGER.log_session(session)
    summary = _summary_md(session if isinstance(session, HumanSession) else None)
    return (
        state,
        "_No active observation._",
        summary,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        "Session abandoned" if summary else "No active session",
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Beer Game Human Baseline") as demo:
        state = gr.State(_empty_state())
        gr.Markdown("# Beer Distribution Game — Human Baseline")
        gr.Markdown(DATA_NOTICE)
        gr.Markdown(TASK_BLURB)

        with gr.Row():
            experience = gr.Radio(
                choices=["yes", "no", "unsure"],
                label="Have you played the beer game before?",
                value=None,
            )
            start_btn = gr.Button("Start game", variant="primary")

        status = gr.Markdown("Answer the question, then start a game.")
        observation = gr.Markdown("_Observation appears after you start._")
        order = gr.Number(
            label="Order quantity (0–128)",
            value=0,
            precision=0,
            minimum=0,
            maximum=128,
            interactive=False,
        )
        with gr.Row():
            order_btn = gr.Button("Place order", interactive=False)
            abandon_btn = gr.Button("Abandon session", interactive=False)

        summary = gr.Markdown()

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
