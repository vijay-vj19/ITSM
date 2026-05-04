import sys
import datetime
import json
import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ITSM Ticket Assistant", page_icon="🛠️", layout="wide")

# Import all logic from app.py
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from app import init as _init, analyse_ticket, prioritize_ticket

# Wrap with Streamlit cache so they only run once per session
init = st.cache_resource(show_spinner="Loading AI models…")(_init)

st.title("ITSM Ticket Assistant")
st.caption("AI-powered ticket analysis and guided resolution")


def render_ai_response(result):
    """Render AI output by extracting description and resolution steps from JSON."""
    parsed = None

    if isinstance(result, (dict, list)):
        parsed = result
    elif isinstance(result, str):
        text = result.strip()
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]

    def _pick_field(payload, candidates):
        if not isinstance(payload, dict):
            return None
        lowered = {str(k).strip().lower(): k for k in payload.keys()}
        for candidate in candidates:
            if candidate in lowered:
                return payload[lowered[candidate]]
        for low_key, original_key in lowered.items():
            if all(token in low_key for token in candidates[0].split()):
                return payload[original_key]
        return None

    def _render_steps(steps):
        if isinstance(steps, list):
            for step in steps:
                st.write(step)
            return
        st.write(steps)

    if isinstance(parsed, dict):
        st.markdown("### AI Response")
        desc = _pick_field(parsed, ["elaborated description", "description", "clear description", "detailed description"])
        steps = _pick_field(parsed, ["resolution steps", "steps", "resolution", "recommended steps"])

        if desc is not None:
            st.markdown("#### Clear Description")
            st.write(desc)

        if steps is not None:
            st.markdown("#### Resolution Steps")
            _render_steps(steps)

        if desc is not None or steps is not None:
            return

    st.markdown("### AI Response")
    st.write(result)


def _prepare_uploaded_df(uploaded_file):
    """Read uploaded Excel and normalize datetime fields for display/analysis."""
    uploaded_df = pd.read_excel(uploaded_file)
    for col in uploaded_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]):
        uploaded_df[col] = uploaded_df[col].dt.strftime("%Y-%m-%d")
    # Uploaded priority is ignored; the app will always assign priority using AI.
    priority_columns = [c for c in uploaded_df.columns if str(c).strip().lower() == "priority"]
    uploaded_df = uploaded_df.drop(columns=priority_columns, errors="ignore")
    return uploaded_df


def _find_column(df, candidates):
    """Find first matching column by exact normalized name or token containment."""
    norm_to_col = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in norm_to_col:
            return norm_to_col[candidate]

    for col in df.columns:
        low = str(col).strip().lower()
        for candidate in candidates:
            parts = candidate.split()
            if all(part in low for part in parts):
                return col
    return None


def _to_text_series(df, columns):
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def _compute_ticket_kpis(df):
    """Compute requested KPI counts using schema-tolerant rules."""
    if df.empty:
        return {
            "redundant": 0,
            "insufficient": 0,
            "sufficient_followup": 0,
            "unassigned": 0,
            "reopened": 0,
        }

    title_col = _find_column(df, ["title", "issue title", "subject", "summary"])
    desc_col = _find_column(df, ["description", "issue description", "details"])
    status_col = _find_column(df, ["status", "state", "ticket status"])
    assignee_col = _find_column(df, ["assignee", "assigned to", "owner", "agent"])
    reopen_col = _find_column(df, ["reopen count", "reopened count", "times reopened"])
    duplicate_col = _find_column(df, ["duplicate", "is duplicate", "redundant"])

    status_text = _to_text_series(df, [status_col] if status_col else [])
    followup_text = _to_text_series(df, [status_col, _find_column(df, ["remarks", "comments", "next action", "notes"])])

    redundant_mask = pd.Series(False, index=df.index)
    if title_col or desc_col:
        keys = _to_text_series(df, [c for c in [title_col, desc_col] if c]).str.replace(r"\s+", " ", regex=True).str.strip()
        redundant_mask = redundant_mask | keys.duplicated(keep="first")
    redundant_mask = redundant_mask | status_text.str.contains(r"duplicate|duplicated|redundant", regex=True, na=False)
    if duplicate_col:
        dup_text = df[duplicate_col].fillna("").astype(str).str.strip().str.lower()
        redundant_mask = redundant_mask | dup_text.str.contains(r"^1$|^true$|^yes$|^y$|duplicate|redundant", regex=True, na=False)

    if desc_col:
        desc_text = df[desc_col].fillna("").astype(str).str.strip().str.lower()
        insufficient_mask = (
            (desc_text.str.len() < 25)
            | desc_text.str.contains(r"^na$|^n/a$|^none$|^unknown$|^not sure$|^help$", regex=True, na=False)
        )
    else:
        insufficient_mask = pd.Series(False, index=df.index)
    insufficient_mask = insufficient_mask | followup_text.str.contains(r"insufficient|missing info|need more info|incomplete", regex=True, na=False)

    needs_followup_mask = followup_text.str.contains(r"follow[ -]?up|pending|waiting|awaiting", regex=True, na=False)
    sufficient_followup_mask = (~insufficient_mask) & needs_followup_mask

    if assignee_col:
        assignee_text = df[assignee_col].fillna("").astype(str).str.strip().str.lower()
        unassigned_mask = assignee_text.eq("") | assignee_text.str.contains(r"unassigned|none|na|n/a|tbd", regex=True, na=False)
    else:
        unassigned_mask = pd.Series(False, index=df.index)

    reopened_mask = status_text.str.contains(r"re-?opened", regex=True, na=False)
    if reopen_col:
        reopen_num = pd.to_numeric(df[reopen_col], errors="coerce").fillna(0)
        reopened_mask = reopened_mask | (reopen_num > 0)

    return {
        "redundant": int(redundant_mask.sum()),
        "insufficient": int(insufficient_mask.sum()),
        "sufficient_followup": int(sufficient_followup_mask.sum()),
        "unassigned": int(unassigned_mask.sum()),
        "reopened": int(reopened_mask.sum()),
    }


@st.cache_data(show_spinner=False)
def _build_mock_dashboard_df(rows=120):
    """Create mock tickets for demo dashboard visuals and KPIs."""
    categories = [
        "Network & Connectivity", "Hardware & Peripherals", "Software & Applications",
        "Email & Communication", "Security & Access", "IT Service Request", "Other"
    ]
    statuses = ["Open", "In Progress", "Resolved", "Reopened"]
    priorities = ["P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"]

    today = datetime.date.today()
    data = []
    for i in range(rows):
        opened_on = today - datetime.timedelta(days=random.randint(0, 29))
        status = random.choices(statuses, weights=[0.35, 0.3, 0.25, 0.1], k=1)[0]
        is_redundant = random.random() < 0.12
        is_insufficient = random.random() < 0.2
        needs_followup = random.random() < 0.28
        is_unassigned = random.random() < 0.16
        is_reopened = status == "Reopened" or random.random() < 0.08

        data.append({
            "Ticket ID": f"MOCK-{1000 + i}",
            "Category": random.choice(categories),
            "Status": status,
            "Priority": random.choice(priorities),
            "Raised On": opened_on.isoformat(),
            "is_redundant": is_redundant,
            "is_insufficient": is_insufficient,
            "needs_followup": needs_followup,
            "is_unassigned": is_unassigned,
            "is_reopened": is_reopened,
        })

    return pd.DataFrame(data)



# ── Load resources ────────────────────────────────────────────────────────────
rails, client, chunks, embeddings = init()
df = _build_mock_dashboard_df()

# ── KPI Snapshot ──────────────────────────────────────────────────────────────
kpis = {
    "redundant": int(df["is_redundant"].sum()),
    "insufficient": int(df["is_insufficient"].sum()),
    "sufficient_followup": int(((~df["is_insufficient"]) & df["needs_followup"]).sum()),
    "unassigned": int(df["is_unassigned"].sum()),
    "reopened": int(df["is_reopened"].sum()),
}
st.subheader("KPI Snapshot")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Redundant Tickets", kpis["redundant"])
k2.metric("Insufficient Information", kpis["insufficient"])
k3.metric("Sufficient but Needs Follow-up", kpis["sufficient_followup"])
k4.metric("Unassigned Tickets", kpis["unassigned"])
k5.metric("Reopened Tickets", kpis["reopened"])

st.caption("Demo dashboard uses generated mock data for presentation only.")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("#### Ticket Quality Mix")
    kpi_chart_df = pd.DataFrame([
        {"KPI": "Redundant", "Count": kpis["redundant"], "Color": "#E4572E"},
        {"KPI": "Insufficient Info", "Count": kpis["insufficient"], "Color": "#F3A712"},
        {"KPI": "Needs Follow-up", "Count": kpis["sufficient_followup"], "Color": "#4C78A8"},
        {"KPI": "Unassigned", "Count": kpis["unassigned"], "Color": "#7A5195"},
        {"KPI": "Reopened", "Count": kpis["reopened"], "Color": "#D45087"},
    ])

    quality_donut = alt.Chart(kpi_chart_df).mark_arc(innerRadius=70).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="KPI", type="nominal", scale=alt.Scale(domain=kpi_chart_df["KPI"].tolist(), range=kpi_chart_df["Color"].tolist()), legend=alt.Legend(title="KPI Type")),
        tooltip=["KPI", "Count"],
    ).properties(height=320)
    st.altair_chart(quality_donut, use_container_width=True)

with chart_col2:
    st.markdown("#### Ticket Volume by Category")
    category_chart_df = df["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
    category_bar = alt.Chart(category_chart_df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Category:N", sort="-x", title="Category"),
        x=alt.X("Count:Q", title="Number of Tickets"),
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
        tooltip=["Category", "Count"],
    ).properties(height=320)
    st.altair_chart(category_bar, use_container_width=True)

st.markdown("#### Daily Ticket Trend (Last 30 Days)")
trend_df = (
    df.groupby("Raised On").size().rename("Tickets").reset_index().sort_values("Raised On")
)
trend_df["Raised On"] = pd.to_datetime(trend_df["Raised On"])
trend_area = alt.Chart(trend_df).mark_area(opacity=0.35, color="#2A9D8F").encode(
    x=alt.X("Raised On:T", title="Date"),
    y=alt.Y("Tickets:Q", title="Tickets Raised"),
    tooltip=[alt.Tooltip("Raised On:T", title="Date"), alt.Tooltip("Tickets:Q", title="Tickets")],
)
trend_line = alt.Chart(trend_df).mark_line(color="#0B6E4F", strokeWidth=3).encode(
    x="Raised On:T",
    y="Tickets:Q",
)
st.altair_chart((trend_area + trend_line).properties(height=300), use_container_width=True)
st.divider()

# ── Upload Excel and Analyse ───────────────────────────────────────────────────
st.subheader("Upload Ticket Excel File")
st.caption("Upload an .xlsx/.xls file, then click a row in the preview table to run AI analysis.")

uploaded_file = st.file_uploader("Add Excel File", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        uploaded_df = _prepare_uploaded_df(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()

    if uploaded_df.empty:
        st.warning("The uploaded file has no rows.")
        st.stop()

    file_token = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("uploaded_file_token") != file_token:
        st.session_state["uploaded_file_token"] = file_token
        st.session_state["last_uploaded_analysis_key"] = None
        st.session_state["last_uploaded_analysis_result"] = None
        st.session_state["last_uploaded_priority"] = None

    st.success("File uploaded successfully")
    table_event = st.dataframe(
        uploaded_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="uploaded_ticket_table",
    )

    selected_rows = table_event.selection.rows if table_event else []
    if not selected_rows:
        st.info("Select any row from the table, then click Start AI Analysis.")
        st.stop()

    selected_row_index = int(selected_rows[0])
    analysis_key = f"{file_token}:{selected_row_index}"
    st.caption(f"Selected row: {selected_row_index + 1}")

    run_analysis = st.button("Start AI Analysis", key="analyse_uploaded_selected")

    if run_analysis:
        selected_ticket = uploaded_df.iloc[selected_row_index].to_dict()

        with st.spinner("Assigning priority with AI…"):
            selected_ticket["Priority"] = prioritize_ticket(selected_ticket, rails)

        with st.spinner("Analysing uploaded ticket…"):
            result = analyse_ticket(selected_ticket, rails, client, chunks, embeddings)

        st.session_state["last_uploaded_analysis_key"] = analysis_key
        st.session_state["last_uploaded_analysis_result"] = result
        st.session_state["last_uploaded_priority"] = selected_ticket["Priority"]

    if st.session_state.get("last_uploaded_analysis_key") == analysis_key and st.session_state.get("last_uploaded_analysis_result") is not None:
        st.success(f"Analysis complete for selected row {selected_row_index + 1}")
        st.info(f"AI-assigned Priority: **{st.session_state.get('last_uploaded_priority', 'P3 - Medium')}**")
        render_ai_response(st.session_state.get("last_uploaded_analysis_result"))
    else:
        st.info("Click Start AI Analysis to generate results for the selected row.")
