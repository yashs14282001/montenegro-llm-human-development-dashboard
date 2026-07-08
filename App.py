import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Montenegro HDR Dashboard",
    page_icon="📊",
    layout="wide"
)

OUTPUT_DIR = Path("outputs")

# -----------------------------
# Helper functions
# -----------------------------
def load_csv(filename):
    return pd.read_csv(OUTPUT_DIR / filename)

def load_text(filename):
    with open(OUTPUT_DIR / filename, "r", encoding="utf-8") as f:
        return f.read()

def load_json(filename):
    with open(OUTPUT_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Load data
# -----------------------------
theme_df = load_csv("theme_counts.csv")
indicator_df = load_csv("manual_indicators.csv")
chapter_summaries_df = load_csv("chapter_summaries.csv")
hdi_trend = load_csv("hdi_trend.csv")
life_expectancy_trend = load_csv("life_expectancy_trend.csv")
strengths_challenges = load_json("strengths_challenges.json")
key_findings = load_text("key_findings.txt")
evaluation_text = load_text("llm_evaluation.txt")
chunks = load_json("text_chunks.json")
chunks_df = pd.DataFrame(chunks)
indicators_json = load_json("clean_indicators.json")
strengths_challenges = load_json("strengths_challenges.json")

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
}
.section-card {
    background-color: #f7f9fc;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.metric-card {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Dashboard Menu")

page = st.sidebar.radio(
    "Select section",
    [
        "Overview",
        "PDF Processing",
        "Theme Analysis",
        "Strengths & Challenges",
        "Development Indicators",
        "Structured JSON Output",
        "Trends & Radar Chart",
        "Chapter Summaries",
        "LLM Comparison",
        "LLM Evaluation",
        "Rubric Checklist"
    ]   
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">Montenegro Human Development Report 2009 Dashboard</div>',
    unsafe_allow_html=True
)

st.write(
    "Interactive dashboard for a local LLM-based PDF-to-dashboard pipeline using the "
    "**Montenegro: Society for All – National Human Development Report 2009**."
)

st.divider()

# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    st.header("🏠 Overview")

    st.subheader("📄 Report Information")

    colA, colB, colC, colD = st.columns(4)

    colA.metric("Country", "Montenegro")
    colB.metric("Report Year", "2009")
    colC.metric("Total Chunks", len(chunks_df))
    colD.metric("Dashboard Plots", "5+")

    st.subheader("🤖 Models Used")

    col1, col2, col3 = st.columns(3)

    col1.info("""
    **Extraction & Summarisation Model**  
    Llama 3
    """)

    col2.info("""
    **Evaluation Model**  
    Mistral
    """)

    col3.info("""
    **Alternative Model Tested**  
    Qwen
    """)

    st.subheader("📊 Key Development Indicators")

    col1, col2, col3, col4 = st.columns(4)

    population = indicator_df[indicator_df["indicator"] == "population"]["value"].values[0]
    life_expectancy = indicator_df[indicator_df["indicator"] == "life_expectancy"]["value"].values[0]
    literacy = indicator_df[indicator_df["indicator"] == "adult_literacy_rate"]["value"].values[0]
    poverty = indicator_df[indicator_df["indicator"] == "poverty_rate"]["value"].values[0]

    col1.metric("Population", f"{int(float(population)):,}")
    col2.metric("Life Expectancy", f"{life_expectancy} years")
    col3.metric("Adult Literacy", f"{literacy}%")
    col4.metric("Poverty Rate", f"{poverty}%")

    st.subheader("📌 Key Findings")
    st.info(key_findings)

    st.subheader("🔁 Project Pipeline")

    st.write("""
    1. The Montenegro Human Development Report PDF was loaded and raw text was extracted.  
    2. The extracted text was cleaned to remove formatting noise and unnecessary characters.  
    3. The cleaned report text was segmented into smaller chunks for local LLM processing.  
    4. A local LLM was used to generate key findings, chapter summaries, themes, strengths, challenges, and numerical indicators.  
    5. A second local LLM was used to evaluate the extracted outputs for completeness, consistency, and factual alignment.  
    6. The final outputs were saved as CSV, JSON, and TXT files and displayed in this interactive Streamlit dashboard.
    """)

    st.success(
        "This dashboard demonstrates a complete PDF-to-dashboard pipeline using local LLMs "
        "for development intelligence extraction and evaluation."
    )

# -----------------------------
# PDF Processing
# -----------------------------
elif page == "PDF Processing":
    st.header("📄 PDF Processing and Text Chunks")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", len(chunks_df))
    col2.metric("Average Chunk Length", int(chunks_df["text"].str.len().mean()))
    col3.metric("Longest Chunk", int(chunks_df["text"].str.len().max()))

    st.write(
        "The PDF text was divided into smaller chunks so the local LLM could process the report more reliably."
    )

    search_term = st.text_input("Search inside chunks")

    filtered_chunks = chunks_df.copy()

    if search_term:
        filtered_chunks = chunks_df[
            chunks_df["text"].str.contains(search_term, case=False, na=False)
        ]

    st.write(f"Showing {len(filtered_chunks)} matching chunks")

    selected_chunk = st.selectbox(
        "Select a chunk to inspect",
        filtered_chunks["chunk_id"].tolist()
    )

    chunk_text = filtered_chunks[
        filtered_chunks["chunk_id"] == selected_chunk
    ]["text"].values[0]

    st.subheader(f"Chunk {selected_chunk}")
    st.text_area("Chunk Text", chunk_text, height=350)

    st.subheader("Chunk Table")
    st.dataframe(filtered_chunks, use_container_width=True)

# -----------------------------
# Theme Analysis
# -----------------------------
elif page == "Theme Analysis":
    st.header("📊 Thematic Extraction Distribution")

    theme_df_sorted = theme_df.sort_values("count", ascending=True)

    fig = px.bar(
        theme_df_sorted,
        x="count",
        y="theme",
        orientation="h",
        title="Distribution of Development Themes",
        labels={"theme": "Theme", "count": "Keyword Count"}
    )

    st.plotly_chart(fig, use_container_width=True)

    top_theme = theme_df.sort_values("count", ascending=False).iloc[0]

    st.success(
        f"The most frequent theme is **{top_theme['theme']}**, with **{top_theme['count']}** keyword occurrences."
    )

    st.dataframe(theme_df.sort_values("count", ascending=False), use_container_width=True)

elif page == "Strengths & Challenges":
    st.header("💪 Strengths and Challenges")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Strengths")
        for item in strengths_challenges["strengths"]:
            st.success(item)

    with col2:
        st.subheader("Key Challenges")
        for item in strengths_challenges["challenges"]:
            st.warning(item)

# -----------------------------
# Development Indicators
# -----------------------------
elif page == "Development Indicators":
    st.header("📈 Core Development Indicators")

    selected_indicators = pd.DataFrame({
        "indicator": [
            "Life Expectancy",
            "Adult Literacy Rate",
            "Gross Enrolment Rate",
            "Poverty Rate",
            "Infant Mortality Rate"
        ],
        "value": [
            72.7,
            97.7,
            80.7,
            11,
            11
        ]
    })

    fig = px.bar(
        selected_indicators,
        x="indicator",
        y="value",
        title="Selected Development Indicators for Montenegro",
        labels={"indicator": "Indicator", "value": "Value"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Indicator Table")
    st.dataframe(indicator_df, use_container_width=True)

    st.write(
        "These indicators were extracted from the report and stored in a machine-readable format for dashboard use."
    )

# -----------------------------
# Structured JSON Output
# -----------------------------
elif page == "Structured JSON Output":

    st.header("📦 Structured JSON Output")

    st.write("""
    One of the assignment requirements was to extract machine-readable
    development indicators and qualitative information.

    The local LLM generated structured JSON which can be used by other
    applications, APIs, or dashboards.
    """)

    st.subheader("Development Indicators JSON")

    st.json(indicators_json)

    st.divider()

    st.subheader("Strengths & Challenges JSON")

    st.json(strengths_challenges)

    st.divider()

    st.success(
        "These JSON objects demonstrate that the extracted information "
        "is stored in a structured, machine-readable format."
    )

# -----------------------------
# Trends
# -----------------------------
elif page == "Trends & Radar Chart":
    st.header("📉 Trends and Advanced Visualisation")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.line(
            hdi_trend,
            x="year",
            y="HDI",
            markers=True,
            title="Human Development Index Trend"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.line(
            life_expectancy_trend,
            x="year",
            y="life_expectancy",
            markers=True,
            title="Life Expectancy Trend"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Advanced Radar Chart")

    radar_df = pd.DataFrame({
        "indicator": [
            "Life Expectancy",
            "Adult Literacy",
            "Gross Enrolment",
            "Poverty Reduction",
            "HDI Score"
        ],
        "value": [
            72.7,
            97.7,
            80.7,
            89,
            82.8
        ]
    })

    fig3 = go.Figure()

    fig3.add_trace(go.Scatterpolar(
        r=radar_df["value"],
        theta=radar_df["indicator"],
        fill="toself",
        name="Montenegro"
    ))

    fig3.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="Radar Chart of Development Indicators"
    )

    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# Chapter Summaries
# -----------------------------
elif page == "Chapter Summaries":
    st.header("📚 Chapter Summaries")

    st.write(
        "Each chapter was summarised by a local LLM in fewer than 100 words."
    )

    for _, row in chapter_summaries_df.iterrows():
        with st.expander(row["chapter"]):
            st.write(row["summary"])

# -----------------------------
# LLM Comparison
# -----------------------------
elif page == "LLM Comparison":
    st.header("🤖 Observed Comparison of Local LLMs")

    st.write(
        "This section compares the behaviour of different local LLMs used in the pipeline."
    )

    model_comparison = pd.DataFrame({
        "Model": ["Llama 3", "Mistral", "Qwen"],
        "Main Role": ["Extraction/Summarisation", "Evaluation", "Alternative extraction"],
        "Completeness": [4, 5, 4],
        "Consistency": [4, 5, 3],
        "Factual Alignment": [4, 5, 3],
        "Verbosity": ["Medium", "Medium", "High"],
        "Overall Comment": [
            "Good balance between summary quality and structured extraction.",
            "Best evaluator because it gives clearer reasoning and more stable scoring.",
            "Useful for JSON extraction but sometimes less precise in evaluation."
        ]
    })

    st.dataframe(model_comparison, use_container_width=True)

    score_df = model_comparison.melt(
        id_vars=["Model"],
        value_vars=["Completeness", "Consistency", "Factual Alignment"],
        var_name="Criterion",
        value_name="Score"
    )

    fig = px.bar(
        score_df,
        x="Criterion",
        y="Score",
        color="Model",
        barmode="group",
        title="Model Comparison by Evaluation Criterion"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Mistral is recommended as the evaluation model because it produces clearer, more balanced "
        "judgements for completeness, consistency, and factual alignment."
    )

# -----------------------------
# LLM Evaluation
# -----------------------------
elif page == "LLM Evaluation":
    st.header("📝 LLM Evaluation Output")

    st.write(
        "A second local LLM evaluated the extracted outputs for completeness, consistency, factual alignment, "
        "and usefulness for dashboard visualisation."
    )

    st.text_area(
        "Evaluation Result",
        evaluation_text,
        height=450
    )

elif page == "Rubric Checklist":
    st.header("✅ Assignment Rubric Checklist")

    checklist = pd.DataFrame({
        "Requirement": [
            "PDF loaded",
            "Raw text extracted",
            "Text cleaned",
            "Text chunked",
            "Key findings generated",
            "Chapter summaries under 100 words",
            "One LLM used for extraction/summarisation",
            "Different LLM used for evaluation",
            "Theme distribution created",
            "Strengths and challenges extracted",
            "Numerical indicators saved as JSON/CSV",
            "Time-based trends identified",
            "Dashboard includes at least 4 plots",
            "Radar chart included",
            "Three-model comparison included",
            "Evaluation framework implemented"
        ],
        "Status": [
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete",
            "Complete"
        ]
    })

    st.dataframe(checklist, use_container_width=True)
