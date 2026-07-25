"""Order quantity display (HTML only — avoids Gradio Number white box)."""

from __future__ import annotations

ORDER_MIN = 0
ORDER_MAX = 128
DEFAULT_ORDER = 4


def clamp_order(quantity: float | int) -> int:
    try:
        qty = int(quantity)
    except (TypeError, ValueError):
        qty = DEFAULT_ORDER
    return max(ORDER_MIN, min(ORDER_MAX, qty))


def nudge_order(quantity: float | int, delta: int) -> int:
    return clamp_order(clamp_order(quantity) + delta)


def order_display_html(qty: int) -> str:
    value = clamp_order(qty)
    return (
        f'<div class="beer-order-num" aria-label="Order quantity">{value}</div>'
    )
