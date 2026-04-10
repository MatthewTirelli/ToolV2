"""Shiny UI layout: top nav tabs, card styling, outputs."""
from __future__ import annotations

from shiny import ui
from shinywidgets import output_widget

DASH_CSS = """
:root {
  --dash-bg: #f5f7fa;
  --card-bg: #ffffff;
  --card-border: #e5e7eb;
  --text: #1e293b;
  --muted: #64748b;
  --shadow: 0 1px 3px rgba(15, 23, 42, 0.08), 0 4px 12px rgba(15, 23, 42, 0.06);
}
body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--dash-bg) !important; color: var(--text); }
.navbar-dash { background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
  padding: 0.85rem 1.5rem; border-radius: 0 0 12px 12px; margin-bottom: 1rem;
  box-shadow: var(--shadow); }
.navbar-dash .brand { font-size: 1.2rem; font-weight: 600; color: #f8fafc; }
.navbar-dash .sub { font-size: 0.8rem; color: #94a3b8; margin-left: 0.75rem; }
.dashboard-card {
  background: var(--card-bg); border-radius: 12px; border: 1px solid var(--card-border);
  padding: 18px 22px; margin-bottom: 1rem; box-shadow: var(--shadow);
  transition: box-shadow 0.2s ease, transform 0.15s ease;
}
.dashboard-card:hover { box-shadow: 0 4px 6px rgba(15, 23, 42, 0.08), 0 8px 20px rgba(15, 23, 42, 0.08); }
.kpi-cards-row { align-items: stretch; }
.kpi-cards-row > [class*="col"] { display: flex; min-width: 0; }
.kpi-card {
  background: var(--card-bg); border-radius: 12px; border: 1px solid var(--card-border);
  padding: 1.1rem 1.25rem; flex: 1; width: 100%; min-height: 7.5rem;
  display: flex; flex-direction: column; justify-content: center;
  box-shadow: var(--shadow); transition: box-shadow 0.2s ease, transform 0.15s ease;
}
.kpi-card:hover { box-shadow: 0 6px 16px rgba(15, 23, 42, 0.1); transform: translateY(-1px); }
.kpi-card-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 600; }
.kpi-card-value { font-size: 1.85rem; font-weight: 700; color: #0f172a; margin-top: 0.35rem; line-height: 1.15; }
.kpi-card-sub { font-size: 0.82rem; color: #94a3b8; margin-top: 0.35rem; }
.ai-panel-inner {
  min-height: 320px; max-height: 420px; overflow-y: auto;
  padding: 1rem 1.1rem; font-size: 0.92rem; line-height: 1.65; color: #475569;
  background: #fafbfc; border-radius: 8px; border: 1px solid #e2e8f0;
}
.section-title { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.75rem; color: #0f172a; }
.tab-content { padding-top: 0.75rem; }
.disclaimer { font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.75rem; }
"""

FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='6' fill='%231e3a5f'/%3E"
    "%3Ctext x='16' y='21' text-anchor='middle' fill='white' font-size='14'%3EM%3C/text%3E"
    "%3C/svg%3E"
)


def app_ui() -> ui.Tag:
    return ui.page_fluid(
        ui.tags.head(
            ui.tags.link(rel="icon", href=FAVICON, type="image/svg+xml"),
            ui.tags.style(DASH_CSS),
        ),
        ui.tags.nav(
            {"class": "navbar-dash"},
            ui.tags.div(
                ui.tags.span({"class": "brand"}, "Measles outbreak risk — US"),
                ui.tags.span({"class": "sub"}, "Situational awareness · CDC data"),
            ),
        ),
        ui.div(
            {"style": "padding: 0 1.25rem 2rem; max-width: 1680px; margin: 0 auto;"},
            ui.p(
                {"class": "disclaimer"},
                "For situational awareness only; not for clinical or policy decisions. Data: CDC.",
            ),
            ui.layout_columns(
                ui.div(
                    ui.input_action_button(
                        "refresh_btn",
                        "Refresh data",
                        class_="btn-primary",
                    ),
                ),
                col_widths=(12,),
            ),
            ui.output_ui("load_status_ui"),
            ui.navset_tab(
                ui.nav_panel(
                    "Overview",
                    ui.output_ui("kpi_cards_ui"),
                    ui.layout_columns(
                        ui.div(
                            {"class": "dashboard-card"},
                            ui.div({"class": "section-title"}, "State risk map"),
                            output_widget("overview_map_plot"),
                        ),
                        ui.div(
                            {"class": "dashboard-card"},
                            ui.div({"class": "section-title"}, "AI report"),
                            ui.div(
                                {"class": "ai-panel-inner"},
                                "AI-generated outbreak insights will appear here.",
                            ),
                        ),
                        col_widths=(7, 5),
                    ),
                    ui.div(
                        {"class": "dashboard-card"},
                        ui.div({"class": "section-title"}, "Baseline risk & model detail"),
                        output_widget("baseline_gauge_plot"),
                        ui.accordion(
                            ui.accordion_panel(
                                "How is alarm probability calculated?",
                                ui.output_ui("alarm_help_ui"),
                            ),
                            ui.accordion_panel(
                                "How is baseline risk/score calculated?",
                                ui.output_ui("baseline_help_ui"),
                            ),
                            id="overview_accordion",
                            open=False,
                        ),
                        ui.download_button("dl_overview", "Download summary CSV", class_="btn-outline-secondary btn-sm mt-2"),
                    ),
                ),
                ui.nav_panel(
                    "Analysis",
                    ui.div(
                        {"class": "dashboard-card"},
                        ui.div({"class": "section-title"}, "Wastewater vs NNDSS filters"),
                        ui.layout_columns(
                            ui.input_select(
                                "ww_year_min",
                                "From year",
                                choices={"": "All"},
                                selected=None,
                            ),
                            ui.input_select(
                                "ww_year_max",
                                "To year",
                                choices={"": "All"},
                                selected=None,
                            ),
                            col_widths=(6, 6),
                        ),
                        ui.div(
                            {"style": "font-size:0.8rem;color:#64748b;margin-top:0.5rem;"},
                            ui.output_ui("ww_info_ui"),
                        ),
                    ),
                    ui.layout_columns(
                        ui.div(
                            {"class": "dashboard-card"},
                            ui.div({"class": "section-title"}, "Wastewater vs NNDSS"),
                            ui.output_ui("ww_chart_ui"),
                            output_widget("ww_analysis_plot"),
                            ui.accordion(
                                ui.accordion_panel(
                                    "Wastewater data audit",
                                    ui.output_ui("ww_audit_ui"),
                                ),
                                id="ww_audit_accordion",
                                open=False,
                            ),
                        ),
                        ui.div(
                            {"class": "dashboard-card"},
                            ui.div({"class": "section-title"}, "Kindergarten MMR coverage"),
                            ui.input_select(
                                "kg_year",
                                "Coverage year",
                                choices={},
                                selected=None,
                            ),
                            output_widget("kg_map_plot"),
                            ui.output_ui("kg_table_ui"),
                            ui.accordion(
                                ui.accordion_panel(
                                    "Kindergarten data audit",
                                    ui.output_ui("kg_audit_ui"),
                                ),
                                id="kg_accordion",
                                open=False,
                            ),
                        ),
                        col_widths=(6, 6),
                    ),
                    ui.div(
                        {"class": "dashboard-card"},
                        ui.div({"class": "section-title"}, "AI Reporter: Wastewater vs NNDSS"),
                        ui.input_action_button("btn_ww_nndss_report", "Generate AI report", class_="btn-primary btn-sm mb-2"),
                        ui.output_ui("ww_nndss_ai_ui"),
                        ui.accordion(
                            ui.accordion_panel(
                                "AI Reporter: understanding wastewater detection",
                                ui.markdown(
                                    "- **Detection frequency** = share of reporting wastewater sites that had measurable measles RNA in that week. It is *not* a count of patients or cases.\n"
                                    "- **Why it matters:** When more sites detect virus, community circulation may be higher; it can sometimes appear in wastewater before confirmed cases are reported.\n"
                                    "- **How to interpret:** An *increase* in detection frequency suggests more sites seeing virus; a *decrease* may mean less circulation or fewer sites reporting.\n"
                                    "- **Lag correlation:** A positive correlation at lag K means detection frequency K weeks ago lines up with cases this week. Correlation does not prove causation; reporting and lab delays affect timing."
                                ),
                            ),
                            id="ww_ai_accordion",
                            open=False,
                        ),
                        ui.p(
                            {"style": "font-size:0.75rem;color:#94a3b8;margin-top:0.5rem;"},
                            "Requires **OLLAMA_API_KEY** in `.env` for AI text. Compares wastewater detection trend vs NNDSS cases.",
                        ),
                    ),
                ),
                ui.nav_panel(
                    "Historical",
                    ui.div(
                        {"class": "dashboard-card"},
                        ui.div({"class": "section-title"}, "National annual measles cases (historical CSV)"),
                        output_widget("hist_annual_plot"),
                    ),
                    ui.div(
                        {"class": "dashboard-card"},
                        ui.div({"class": "section-title"}, "NNDSS weekly (national)"),
                        ui.input_select(
                            "nndss_view",
                            "NNDSS weekly view",
                            choices={"104": "Last 104 weeks", "all": "All available weeks"},
                            selected="104",
                        ),
                        output_widget("nndss_weekly_plot"),
                        ui.output_ui("nndss_recent_table_ui"),
                        ui.accordion(
                            ui.accordion_panel(
                                "NNDSS data audit",
                                ui.output_ui("nndss_audit_ui"),
                            ),
                            id="nndss_accordion",
                            open=False,
                        ),
                    ),
                ),
                ui.nav_panel(
                    "State & risk",
                    ui.div(
                        {"class": "dashboard-card"},
                        ui.div({"class": "section-title"}, "State-level risk map"),
                        output_widget("state_tab_map_plot"),
                    ),
                    ui.div(
                        {"class": "dashboard-card"},
                        ui.output_ui("state_table_ui"),
                        ui.accordion(
                            ui.accordion_panel(
                                "How state risk is calculated",
                                ui.output_ui("state_risk_help_ui"),
                            ),
                            ui.accordion_panel(
                                "What is the wastewater signal?",
                                ui.markdown(
                                    "The **wastewater signal** is a measure of how much measles virus genetic material was detected "
                                    "in wastewater (sewage) for that state over the last 4 weeks. The number is normalized for flow and "
                                    "population so states can be compared. In the table above it is shown **per 100k** (e.g. 2.48 = 248,000 in raw units). "
                                    "**Higher values** mean more measles signal in wastewater in that period; **lower values** mean less. "
                                    "States with no wastewater monitoring in the dataset show **No coverage**. "
                                    "The signal comes from CDC wastewater surveillance (WastewaterScan) and is used here as one input to state risk."
                                ),
                            ),
                            id="state_accordion",
                            open=False,
                        ),
                    ),
                    ui.div(
                        {"class": "dashboard-card"},
                        ui.div({"class": "section-title"}, "AI report for a state"),
                        ui.input_select(
                            "state_report_select",
                            "Choose a state",
                            choices={"": "— Select a state —"},
                            selected="",
                        ),
                        ui.input_action_button("btn_state_report", "Generate AI report for this state", class_="btn-primary btn-sm"),
                        ui.output_ui("state_report_ai_ui"),
                    ),
                ),
                ui.nav_panel(
                    "Forecast",
                    ui.div(
                        {"class": "dashboard-card"},
                        ui.div({"class": "section-title"}, "Forecast by state"),
                        ui.output_ui("forecast_table_ui"),
                        ui.output_ui("forecast_national_ui"),
                        ui.input_action_button("forecast_ai_btn", "Generate AI interpretation", class_="btn-primary btn-sm me-2"),
                        ui.input_action_button("forecast_ai_regen", "Regenerate", class_="btn-outline-secondary btn-sm"),
                        ui.output_ui("forecast_ai_ui"),
                    ),
                ),
                id="main_tabs",
            ),
        ),
        title="Measles outbreak risk — US",
    )
