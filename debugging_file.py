import streamlit as st
import os
from dotenv import load_dotenv
import json
from enum import Enum

# --- LangChain Imports (we keep these) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional

# --- Import Data Tools (we keep this) ---
# This will still print the "Data tools initialized" message if successful
from samarth_app.data_tools import (
    get_state_comparison_data,
    find_district_production_extrema,
    get_trend_analysis_data,
    get_policy_analysis_data,
)

# --- Load Environment ---
load_dotenv()

# --- Streamlit UI (runs first) ---
st.set_page_config(page_title="Project Samarth - DEBUG", page_icon="🐛")
st.title("🐛 Project Samarth - Debug Mode")
st.subheader("An Intelligent Q&A System for India's Agricultural & Climate Data")

st.success("If you can see this message, Streamlit is working correctly!")
st.info("This debug version does not connect to the AI models on startup. The main logic is disabled.")

# --- All AI setup and logic is disabled below ---

# This is a placeholder for the user input
user_question = st.text_area(
    "Ask a complex question (Logic is currently disabled):",
    placeholder="e.g., Identify the district in Maharashtra with the highest production of Jowar in 2014...",
    height=125,
    disabled=True # Disable the text area in debug mode
)

if st.button("Get Answer (Disabled)"):
    st.warning("The AI logic is disabled in this debug file.")

# --- Check for API Key ---
st.header("Initial Setup Checks")
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Check Failed: Google API key was NOT found in the .env file.")
else:
    st.success("Check Passed: Found the GOOGLE_API_KEY in your .env file.")

