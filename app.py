import streamlit as st
import os
from dotenv import load_dotenv
import json
from enum import Enum

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional

# --- Import ALL Our Specialist Data Tools ---
from samarth_app.data_tools import (
    get_state_comparison_data,
    find_district_production_extrema,
    get_trend_analysis_data,
    get_policy_analysis_data,
)

# --- Load Environment ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Google API key not found in the .env file. Please add it to run the application.")
    st.stop()

# --- ARCHITECTURE SETUP: THE ROUTER (with Few-Shot Examples for Accuracy) ---
class QueryType(str, Enum):
    STATE_COMPARISON = "state_comparison"
    DISTRICT_EXTREMA = "district_extrema"
    TREND_ANALYSIS = "trend_analysis"
    POLICY_ADVICE = "policy_advice"
    UNKNOWN = "unknown"

class RouteQuery(BaseModel):
    query_type: QueryType = Field(..., description="The type of query the user is asking.")

router_llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)
# This prompt includes examples to make the router much more accurate.
router_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are an expert at routing a user's query. Classify it into one of the following categories.

Here are some examples to guide you:

- User Query: "Compare the rainfall and top 3 cereals in Gujarat and Rajasthan for 2011."
- Classification: `{QueryType.STATE_COMPARISON.value}`

- User Query: "Identify the district in Uttar Pradesh with the highest production of Wheat and compare with the lowest in Punjab for the same year."
- Classification: `{QueryType.DISTRICT_EXTREMA.value}`

- User Query: "Analyze the production trend of Pulses in Maharashtra over the last decade."
- Classification: `{QueryType.TREND_ANALYSIS.value}`

- User Query: "What are the data-backed arguments to promote Bajra over Sugarcane?"
- Classification: `{QueryType.POLICY_ADVICE.value}`
"""),
    ("human", "{query}")
])
router_chain = router_prompt | router_llm.with_structured_output(RouteQuery)


# --- SPECIALIST DEFINITIONS (Parsers and Formatters) ---

# Specialist 1: State Comparison
class StateInput(BaseModel):
    states: List[str]; year: int; top_n: int; crop_type: Optional[str] = None
parser_state_chain = (ChatPromptTemplate.from_messages([("system", "Extract the list of states, the single year, the number of top crops, and an optional crop type from the user's query."), ("human", "{query}")]) | router_llm.with_structured_output(StateInput))
def format_state_comparison(json_string):
    data = json.loads(json_string)
    report = "### State-Level Comparison\n"
    for state, details in data.items():
        if state == "data_sources_used": continue
        report += f"\n**Results for {state.title()}:**\n- **Normal Annual Rainfall:** {details.get('normal_annual_rainfall_mm', 'N/A')} mm\n- **Top Crops by Production:**\n"
        if details.get('top_crops'):
            for crop in details['top_crops']: report += f"  - {crop.get('Crop', '').title()}: {int(crop.get('Production', 0)):,} tonnes\n"
        else: report += "  - No crop data found for the specified criteria.\n"
    if "data_sources_used" in data:
        report += "\n---\n*Sources:*\n"; [report := report + f"- *{s}*\n" for s in data["data_sources_used"]]
    return report

# Specialist 2: District Extrema
class DistrictInput(BaseModel):
    state_1: str; crop_1: str; state_2: str; crop_2: str; year: int
parser_district_chain = (ChatPromptTemplate.from_messages([("system", "Extract the two states, the specific crop, and the single year."), ("human", "{query}")]) | router_llm.with_structured_output(DistrictInput))
def format_district_comparison(h_json, l_json):
    h, l = json.loads(h_json), json.loads(l_json)
    if "error" in h or "error" in l: return f"Error:\n- Highest: {h.get('error', 'N/A')}\n- Lowest: {l.get('error', 'N/A')}"
    report = f"""### District-Level Production Comparison\nFor **{h['year']}**:\n- The district with the **highest** production of **{h['crop']}** in **{h['state']}** was **{h['district']}** ({h['production_tonnes']:,} tonnes).\n- The district with the **lowest** (non-zero) production of **{l['crop']}** in **{l['state']}** was **{l['district']}** ({l['production_tonnes']:,} tonnes)."""
    if "data_sources_used" in h:
        report += "\n\n---\n*Source:*\n"; [report := report + f"- *{s}*\n" for s in h["data_sources_used"]]
    return report

# Specialist 3: Trend Analysis
class TrendInput(BaseModel):
    region: str; crop_type: str; start_year: int; end_year: int
parser_trend_chain = (ChatPromptTemplate.from_messages([("system", "Extract the region, crop type, start and end year for the trend analysis."), ("human", "{query}")]) | router_llm.with_structured_output(TrendInput))
def format_trend_analysis(json_string):
    data = json.loads(json_string)
    if "error" in data: return f"Error: {data['error']}"
    report = f"### Production Trend for {data['crop_type']} in {data['region']} ({data['period']})\n"
    for item in data.get('production_trend_tonnes', []): report += f"- **{item['Crop_Year']}**: {int(item['Production']):,} tonnes\n"
    report += f"\n**Correlation Context:**\n- The normal annual rainfall for this region is **{data.get('normal_annual_rainfall_mm_for_correlation', 'N/A')} mm**. *({data.get('note', '')})*"
    if "data_sources_used" in data:
        report += "\n\n---\n*Sources:*\n"; [report := report + f"- *{s}*\n" for s in data["data_sources_used"]]
    return report

# Specialist 4: Policy Advice
class PolicyInput(BaseModel):
    region: str; crop_a: str; crop_b: str; years: int
parser_policy_chain = (ChatPromptTemplate.from_messages([("system", "Extract the region, the two crops for comparison, and the number of years from the policy query."), ("human", "{query}")]) | router_llm.with_structured_output(PolicyInput))
def synthesize_arguments(evidence_json: str, crop_a: str, crop_b: str, region: str) -> str:
    """Takes raw data and uses an LLM to generate reasoned arguments."""
    s_llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0.3)
    s_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert policy advisor. Your task is to use the provided data to generate three distinct, compelling, data-backed arguments to support a policy decision. Frame the arguments clearly and concisely."),
        ("human", """**Policy Proposal:** Promote the cultivation of **{crop_a}** over **{crop_b}** in **{region}**.
        \n**Supporting Data:**\n```json\n{evidence}\n```\nPlease generate the three most compelling, data-backed arguments based *only* on the data provided.""")
    ])
    s_chain = s_prompt | s_llm
    response = s_chain.invoke({"crop_a": crop_a, "crop_b": crop_b, "region": region, "evidence": evidence_json})
    return response.content


# --- Streamlit UI ---
st.set_page_config(page_title="Project Samarth", page_icon="🌾")
st.title("🌾 Project Samarth")
st.subheader("An Intelligent Q&A System for India's Agricultural & Climate Data")

st.info("""
**Ask a complex question about India's agriculture and climate.** The system will automatically identify the question type and perform the correct analysis.
""")

user_question = st.text_area(
    "Ask your question:",
    placeholder="e.g., Identify the district in Maharashtra with the highest production of Jowar in 2014 and compare that with the district with the lowest production of Jowar in Karnataka for the same year.",
    height=125,
)

if st.button("Get Answer"):
    if user_question:
        try:
            with st.spinner("1. Routing query to the correct analytical tool..."):
                route = router_chain.invoke({"query": user_question})
                # REMOVED: Commented out the info message for a cleaner UI
                # st.info(f"Query identified as: **{route.query_type.name}**. Executing analysis...")

            with st.spinner("2. Parsing query and fetching data..."):
                if route.query_type == QueryType.STATE_COMPARISON:
                    args = parser_state_chain.invoke({"query": user_question})
                    json_data = get_state_comparison_data(states=args.states, year=args.year, top_n=args.top_n, crop_type=args.crop_type)
                    report = format_state_comparison(json_data)
                    st.markdown(report)

                elif route.query_type == QueryType.DISTRICT_EXTREMA:
                    args = parser_district_chain.invoke({"query": user_question})
                    h_json = find_district_production_extrema(state=args.state_1, year=args.year, crop=args.crop_1, find='highest')
                    l_json = find_district_production_extrema(state=args.state_2, year=args.year, crop=args.crop_1, find='lowest') # Use crop_1 for both
                    report = format_district_comparison(h_json, l_json)
                    st.markdown(report)
                
                elif route.query_type == QueryType.TREND_ANALYSIS:
                    args = parser_trend_chain.invoke({"query": user_question})
                    json_data = get_trend_analysis_data(region=args.region, crop_type=args.crop_type, start_year=args.start_year, end_year=args.end_year)
                    report = format_trend_analysis(json_data)
                    st.markdown(report)

                elif route.query_type == QueryType.POLICY_ADVICE:
                    args = parser_policy_chain.invoke({"query": user_question})
                    evidence_json = get_policy_analysis_data(region=args.region, crop_a=args.crop_a, crop_b=args.crop_b, years=args.years)
                    with st.spinner("3. Synthesizing data-backed arguments..."):
                        final_report = synthesize_arguments(evidence_json, crop_a=args.crop_a, crop_b=args.crop_b, region=args.region)
                        st.markdown("### Policy Recommendation Arguments")
                        st.markdown(final_report)

                else: # Handles UNKNOWN
                    st.warning("I am currently equipped to handle four types of queries: State Comparisons, District Comparisons, Trend Analysis, and Policy Advice. Please try rephrasing your question to fit one of these formats.")
        except Exception as e:
            st.error(f"A critical error occurred: {e}")
            # Optionally add more detailed logging for debugging
            # import traceback
            # st.error(traceback.format_exc())
    else:
        st.warning("Please enter a question.")

