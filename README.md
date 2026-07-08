# Montenegro Human Development Report 2009: Local LLM PDF-to-Dashboard Pipeline

This project builds a complete local LLM-based AI system for extracting structured development indicators and insights from the **Montenegro National Human Development Report 2009**.

The system processes a real-world UN Human Development Report PDF, extracts and cleans text, segments the document into chunks, uses local large language models for summarisation and structured information extraction, evaluates the outputs using a second LLM, and presents the results in an interactive Streamlit dashboard.

## Project Objective

The objective of this assignment is to build and evaluate a PDF-to-dashboard pipeline using local LLMs. The dashboard highlights key development indicators, thematic distributions, chapter summaries, strengths and challenges, trends, and model comparison outputs.

## Main Features

- PDF text extraction using PyMuPDF
- Text cleaning and chunking
- Local LLM-based summarisation
- Chapter summaries under 100 words
- Thematic extraction across key development themes
- Structured numerical indicator extraction as JSON
- Strengths and challenges extraction
- LLM-based evaluation of extracted outputs
- Interactive Streamlit dashboard
- Theme distribution visualisation
- Development indicator plots
- HDI and life expectancy trend plots
- Radar chart for advanced visualisation
- Cross-LLM behaviour comparison

## Models Used

| Purpose | Model |
|---|---|
| Extraction and summarisation | Llama 3 |
| Evaluation | Mistral |
| Alternative model tested | Qwen |

## Dashboard Sections

The Streamlit dashboard includes:

1. Overview  
2. PDF Processing and Text Chunks  
3. Theme Analysis  
4. Strengths and Challenges  
5. Development Indicators  
6. Structured JSON Output  
7. Trends and Radar Chart  
8. Chapter Summaries  
9. LLM Comparison  
10. LLM Evaluation  
11. Rubric Checklist  

## Project Structure

```text
.
├── assignment-1.ipynb
├── App.py
├── requirements.txt
├── README.md
├── montenegronhdr2009en.pdf
└── outputs/
    ├── theme_counts.csv
    ├── manual_indicators.csv
    ├── clean_indicators.json
    ├── strengths_challenges.json
    ├── chapter_summaries.csv
    ├── hdi_trend.csv
    ├── life_expectancy_trend.csv
    ├── key_findings.txt
    ├── llm_evaluation.txt
    └── text_chunks.json
