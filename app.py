import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import json
import io
import re

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with your CSV",
    page_icon="📊",
    layout="wide"
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message-user {
        background: #e8f4fd;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #1f77b4;
    }
    .chat-message-assistant {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #28a745;
    }
    .stat-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ── Groq client ───────────────────────────────────────────────────────────────
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to Streamlit secrets.")
        st.stop()
    return Groq(api_key=api_key)


# ── Data summary for LLM context ─────────────────────────────────────────────
def get_data_summary(df: pd.DataFrame) -> str:
    summary = []
    summary.append(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    summary.append(f"Columns: {', '.join(df.columns.tolist())}")
    summary.append("\nColumn types and sample values:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        if df[col].dtype in ['int64', 'float64']:
            summary.append(
                f"  - {col} ({dtype}): min={df[col].min():.2f}, "
                f"max={df[col].max():.2f}, mean={df[col].mean():.2f}, "
                f"nulls={nulls}"
            )
        else:
            unique = df[col].nunique()
            sample = df[col].dropna().head(3).tolist()
            summary.append(
                f"  - {col} ({dtype}): {unique} unique values, "
                f"sample={sample}, nulls={nulls}"
            )
    summary.append(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    return "\n".join(summary)


# ── Chart detector ────────────────────────────────────────────────────────────
def detect_chart_intent(question: str) -> bool:
    chart_keywords = [
        "chart", "plot", "graph", "visuali", "show me",
        "display", "bar", "line", "pie", "scatter", "histogram",
        "distribution", "trend", "compare", "breakdown"
    ]
    q = question.lower()
    return any(k in q for k in chart_keywords)


# ── LLM call ─────────────────────────────────────────────────────────────────
def ask_groq(client, question: str, data_summary: str, df: pd.DataFrame, chat_history: list) -> dict:
    want_chart = detect_chart_intent(question)

    system_prompt = f"""You are a data analyst assistant. The user has uploaded a CSV file.

DATA SUMMARY:
{data_summary}

Your job:
1. Answer the user's question about their data clearly and concisely.
2. Use actual values from the data summary to give specific answers.
3. If the user asks for a chart or visualisation, respond with a JSON block like this:

```json
{{
  "chart_type": "bar" | "line" | "pie" | "scatter" | "histogram",
  "x": "column_name",
  "y": "column_name_or_null",
  "title": "Chart title",
  "color": "column_name_or_null",
  "aggregation": "sum" | "mean" | "count" | "none"
}}
```

4. Always give a text explanation alongside any chart.
5. If asked about data quality, mention nulls, duplicates, or outliers.
6. Keep answers focused and helpful. Do not make up data not in the summary.
"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in chat_history[-6:]:  # last 3 turns for context
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content
    chart_config = None

    # extract JSON chart config if present
    json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if json_match:
        try:
            chart_config = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
        # remove json block from text
        raw = re.sub(r"```json\s*.*?\s*```", "", raw, flags=re.DOTALL).strip()

    return {"text": raw, "chart_config": chart_config}


# ── Chart renderer ────────────────────────────────────────────────────────────
def render_chart(df: pd.DataFrame, config: dict):
    try:
        chart_type = config.get("chart_type", "bar")
        x = config.get("x")
        y = config.get("y")
        title = config.get("title", "Chart")
        color = config.get("color")
        aggregation = config.get("aggregation", "none")

        # validate columns exist
        for col in [x, y, color]:
            if col and col not in df.columns:
                st.warning(f"Column '{col}' not found. Skipping chart.")
                return

        # aggregate if needed
        plot_df = df.copy()
        if aggregation != "none" and x and y:
            if aggregation == "sum":
                plot_df = df.groupby(x)[y].sum().reset_index()
            elif aggregation == "mean":
                plot_df = df.groupby(x)[y].mean().reset_index()
            elif aggregation == "count":
                plot_df = df.groupby(x)[y].count().reset_index()

        if chart_type == "bar":
            fig = px.bar(plot_df, x=x, y=y, title=title, color=color,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        elif chart_type == "line":
            fig = px.line(plot_df, x=x, y=y, title=title, color=color,
                          markers=True)
        elif chart_type == "pie":
            fig = px.pie(plot_df, names=x, values=y, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x, y=y, title=title, color=color,
                             trendline="ols" if y else None)
        elif chart_type == "histogram":
            fig = px.histogram(plot_df, x=x, title=title, color=color,
                               nbins=30)
        else:
            st.warning(f"Unknown chart type: {chart_type}")
            return

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Arial", size=13),
            title_font_size=16,
            margin=dict(t=50, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not render chart: {e}")


# ── Quick stats ───────────────────────────────────────────────────────────────
def show_quick_stats(df: pd.DataFrame):
    cols = st.columns(4)
    numeric_cols = df.select_dtypes(include="number").columns
    with cols[0]:
        st.markdown(f"""<div class="stat-card">
            <h3>{df.shape[0]:,}</h3><p>Rows</p></div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""<div class="stat-card">
            <h3>{df.shape[1]}</h3><p>Columns</p></div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""<div class="stat-card">
            <h3>{df.isnull().sum().sum()}</h3><p>Missing Values</p></div>""", unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""<div class="stat-card">
            <h3>{len(numeric_cols)}</h3><p>Numeric Columns</p></div>""", unsafe_allow_html=True)


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-header">📊 Chat with your CSV</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload any spreadsheet and ask questions in plain English. Get answers and charts instantly.</div>', unsafe_allow_html=True)

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "data_summary" not in st.session_state:
        st.session_state.data_summary = None

    # sidebar
    with st.sidebar:
        st.header("Upload your file")
        uploaded_file = st.file_uploader(
            "CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Max 200MB"
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.df = df
                st.session_state.data_summary = get_data_summary(df)
                st.session_state.chat_history = []
                st.success(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        st.divider()
        st.markdown("**Example questions:**")
        examples = [
            "What are the column names?",
            "Show me a bar chart of sales by region",
            "What is the average value of [column]?",
            "Are there any missing values?",
            "Show the distribution of [column]",
            "Which category has the highest total?",
        ]
        for ex in examples:
            st.markdown(f"- *{ex}*")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # main content
    if st.session_state.df is None:
        st.info("Upload a CSV or Excel file from the sidebar to get started.")
        st.markdown("### What can you ask?")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Data exploration**\n- How many rows?\n- What columns exist?\n- Any missing values?")
        with col2:
            st.markdown("**Analysis**\n- What is the average X?\n- Which category has the most Y?\n- Show me the top 5...")
        with col3:
            st.markdown("**Visualisation**\n- Plot sales over time\n- Show a pie chart of categories\n- Bar chart of X by Y")
        return

    df = st.session_state.df

    # data preview
    with st.expander("Data preview", expanded=False):
        show_quick_stats(df)
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()

    # chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("chart_config"):
                render_chart(df, msg["chart_config"])

    # input
    question = st.chat_input("Ask anything about your data...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.markdown(f'<div class="chat-message-user">👤 {question}</div>', unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            client = get_groq_client()
            result = ask_groq(
                client,
                question,
                st.session_state.data_summary,
                df,
                st.session_state.chat_history
            )

        st.markdown(f'<div class="chat-message-assistant">🤖 {result["text"]}</div>', unsafe_allow_html=True)

        if result["chart_config"]:
            render_chart(df, result["chart_config"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["text"],
            "chart_config": result.get("chart_config")
        })


if __name__ == "__main__":
    main()
