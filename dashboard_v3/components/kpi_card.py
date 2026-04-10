"""Reusable KPI card markup for Shiny (card-based metrics)."""
from __future__ import annotations

from typing import Optional

from shiny import ui


def kpi_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    *,
    card_id: Optional[str] = None,
) -> ui.Tag:
    """White card: small title, large value, optional subtitle (delta / context)."""
    extra: dict = {"class": "kpi-card"}
    if card_id:
        extra["id"] = card_id
    children = [
        ui.div({"class": "kpi-card-title"}, title),
        ui.div({"class": "kpi-card-value"}, value),
    ]
    if subtitle:
        children.append(ui.div({"class": "kpi-card-sub"}, subtitle))
    return ui.div(extra, *children)


def kpi_row(*cards: ui.Tag) -> ui.Tag:
    """Responsive row of equal-height KPI cards."""
    return ui.div(
        {"class": "row g-3 kpi-cards-row"},
        *[ui.div({"class": "col-lg-4 col-md-6"}, c) for c in cards],
    )
