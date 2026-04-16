import sys
import datetime
import json
from pathlib import Path

import streamlit as st

# Import all logic from app.py
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from app import init as _init, load_tickets as _load_tickets, analyse_ticket

# Wrap with Streamlit cache so they only run once per session
init = st.cache_resource(show_spinner="Loading AI models…")(_init)
load_tickets = st.cache_data(show_spinner="Loading tickets…")(_load_tickets)


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



# ── Load resources ────────────────────────────────────────────────────────────
rails, client, chunks, embeddings = init()
df = load_tickets()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Existing Tickets", "➕ Submit New Ticket"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Existing Tickets
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Ticket List")

    # Display columns (subset for readability)
    display_cols = [c for c in ["Ticket ID", "Name", "Title", "Category", "Priority", "Status", "Raised On"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Analyse a Ticket")

    id_col = next((c for c in ["Ticket ID", "ID", "id"] if c in df.columns), df.columns[0])
    ticket_ids = df[id_col].astype(str).tolist()
    selected_id = st.selectbox("Select Ticket ID", ticket_ids)

    selected_row = df[df[id_col].astype(str) == selected_id].iloc[0].to_dict()

    with st.expander("Ticket Details", expanded=True):
        col1, col2 = st.columns(2)
        items = list(selected_row.items())
        mid = len(items) // 2
        for key, value in items[:mid]:
            col1.markdown(f"**{key}:** {value}")
        for key, value in items[mid:]:
            col2.markdown(f"**{key}:** {value}")

    if st.button("🤖 Run AI Analysis", key="analyse_existing"):
        with st.spinner("Analysing ticket…"):
            result = analyse_ticket(selected_row, rails, client, chunks, embeddings)
        st.success("Analysis complete")
        render_ai_response(result)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Submit New Ticket
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Submit a New Ticket")

    categories = [
        "Network & Connectivity", "Hardware & Peripherals", "Software & Applications",
        "Email & Communication", "Security & Access", "IT Service Request", "Other"
    ]
    priorities = ["P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"]

    with st.form("new_ticket_form"):
        col1, col2 = st.columns(2)
        name = col1.text_input("Full Name *")
        email = col2.text_input("Email Address *")
        title = st.text_input("Issue Title *")
        description = st.text_area("Description *", height=150)
        col3, col4 = st.columns(2)
        category = col3.selectbox("Category", categories)
        priority = col4.selectbox("Priority", priorities)
        submitted = st.form_submit_button("Submit Ticket")

    if submitted:
        missing = [f for f, v in [("Name", name), ("Email", email), ("Title", title), ("Description", description)] if not v.strip()]
        if missing:
            st.error(f"Please fill in: {', '.join(missing)}")
        else:
            new_ticket = {
                "Ticket ID": f"INC{datetime.datetime.now().strftime('%H%M%S')}",
                "Name": name,
                "Email": email,
                "Title": title,
                "Description": description,
                "Category": category,
                "Priority": priority,
                "Raised On": datetime.date.today().isoformat(),
            }

            st.success(f"Ticket **{new_ticket['Ticket ID']}** submitted!")

            with st.expander("Your Ticket", expanded=False):
                for key, value in new_ticket.items():
                    st.markdown(f"**{key}:** {value}")

            with st.spinner("Generating AI response…"):
                result = analyse_ticket(new_ticket, rails, client, chunks, embeddings)
            render_ai_response(result)
