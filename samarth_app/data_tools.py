import pandas as pd
import os
import json

# --- File Paths and Data Loading ---
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    processed_data_path = os.path.join(project_root, 'data', 'processed')

    crop_df = pd.read_csv(os.path.join(processed_data_path, 'crop_production_cleaned.csv'))
    rain_df = pd.read_csv(os.path.join(processed_data_path, 'rainfall_cleaned.csv'))
    print("✅ Data tools initialized: Cleaned datasets loaded into memory.")
except FileNotFoundError:
    # This provides a fallback if the script is run in an unexpected environment.
    crop_df = pd.DataFrame()
    rain_df = pd.DataFrame()

# --- Intelligence Layer: Mappings and Constants ---
CROP_TYPE_MAP = {
    'PULSES': ['ARHAR/TUR', 'GRAM', 'MOONG(GREEN GRAM)', 'URAD', 'OTHER KHARIF PULSES', 'OTHER RABI PULSES', 'MASOOR', 'PEAS & BEANS (PULSES)'],
    'OILSEEDS': ['GROUNDNUT', 'SESAMUM', 'RAPESEED & MUSTARD', 'SUNFLOWER', 'SOYABEAN', 'NIGER SEED', 'CASTOR-SEED', 'LINSEED', 'SAFFLOWER'],
    'CEREALS': ['RICE', 'WHEAT', 'MAIZE', 'BAJRA', 'JOWAR', 'RAGI', 'BARLEY', 'SMALL MILLETS'],
}
DATA_SOURCES = [
    "Crop Production Data: Directorate of Economics and Statistics, Ministry of Agriculture & Farmers Welfare.",
    "Rainfall Data: India Meteorological Department (IMD) - District Wise Rainfall Normals."
]

# --- Specialist Tool 1: State-Level Comparison ---
def get_state_comparison_data(states: list[str], year: int, top_n: int = 5, crop_type: str = None) -> str:
    """Fetches and compares rainfall and top crops for a list of states in a single year."""
    results = {}
    for state in states:
        state_upper = state.strip().upper()
        state_data = {}
        # Rainfall
        rain_state_df = rain_df[rain_df['State_Name'] == state_upper]
        state_data['normal_annual_rainfall_mm'] = round(rain_state_df['ANNUAL'].mean(), 2) if not rain_state_df.empty else "N/A"
        # Crops
        crop_state_year_df = crop_df[(crop_df['State_Name'] == state_upper) & (crop_df['Crop_Year'] == year)]
        if crop_type and crop_type.upper() in CROP_TYPE_MAP:
            crop_list = CROP_TYPE_MAP[crop_type.upper()]
            crop_state_year_df = crop_state_year_df[crop_state_year_df['Crop'].isin(crop_list)]
        
        if not crop_state_year_df.empty:
            top_crops = (crop_state_year_df.groupby('Crop')['Production'].sum().nlargest(top_n).reset_index())
            state_data['top_crops'] = top_crops.to_dict('records')
        else:
            state_data['top_crops'] = []
        results[state] = state_data
    results["data_sources_used"] = DATA_SOURCES
    return json.dumps(results, indent=2)

# --- Specialist Tool 2: District-Level Extrema ---
def find_district_production_extrema(state: str, year: int, crop: str, find: str) -> str:
    """Finds the district with the highest or lowest production of a specific crop."""
    if crop_df.empty: return json.dumps({"error": "Crop data not loaded."})
    state_upper, crop_upper = state.strip().upper(), crop.strip().upper()
    df = crop_df[(crop_df['State_Name'] == state_upper) & (crop_df['Crop_Year'] == year) & (crop_df['Crop'] == crop_upper)]
    if df.empty: return json.dumps({"error": f"No data for '{crop}' in '{state}' for {year}."})
    
    result_df = None
    if find == 'highest':
        result_df = df.loc[df['Production'].idxmax()]
    elif find == 'lowest':
        non_zero_df = df[df['Production'] > 0]
        if not non_zero_df.empty:
            result_df = non_zero_df.loc[non_zero_df['Production'].idxmin()]
    
    if result_df is None or result_df.empty:
        return json.dumps({"error": f"Could not find a {find} (non-zero) production district."})
    
    return json.dumps({
        "state": state, "district": result_df['District_Name'].title(), "crop": result_df['Crop'].title(),
        "year": int(result_df['Crop_Year']), "production_tonnes": int(result_df['Production']),
        "data_sources_used": [DATA_SOURCES[0]]
    })

# --- Specialist Tool 3: Trend Analysis ---
def get_trend_analysis_data(region: str, crop_type: str, start_year: int, end_year: int) -> str:
    """Analyzes production trends for a crop type in a region over a range of years."""
    region_upper = region.strip().upper()
    df = crop_df[(crop_df['State_Name'] == region_upper) & (crop_df['Crop_Year'].between(start_year, end_year))]
    
    if crop_type and crop_type.upper() in CROP_TYPE_MAP:
        df = df[df['Crop'].isin(CROP_TYPE_MAP[crop_type.upper()])]
    
    if df.empty: return json.dumps({"error": f"No data found for '{crop_type}' in '{region}' between {start_year}-{end_year}."})
    
    trend = df.groupby('Crop_Year')['Production'].sum().reset_index().to_dict('records')
    
    rain_df_region = rain_df[rain_df['State_Name'] == region_upper]
    normal_rainfall = round(rain_df_region['ANNUAL'].mean(), 2) if not rain_df_region.empty else "N/A"

    return json.dumps({
        "analysis_type": "Production Trend Analysis",
        "region": region, "crop_type": crop_type, "period": f"{start_year}-{end_year}",
        "production_trend_tonnes": trend,
        "normal_annual_rainfall_mm_for_correlation": normal_rainfall,
        "note": "Rainfall data is a long-term average and not year-specific.",
        "data_sources_used": DATA_SOURCES
    }, indent=2)

# --- Specialist Tool 4: Policy Analysis Data Gatherer ---
def get_policy_analysis_data(region: str, crop_a: str, crop_b: str, years: int) -> str:
    """Gathers data for a policy comparison between two crops in a region over N years."""
    if crop_df.empty: return json.dumps({"error": "Crop data not loaded."})

    latest_year = int(crop_df['Crop_Year'].max())
    start_year = latest_year - years + 1

    region_upper = region.strip().upper()
    crops_upper = [crop_a.strip().upper(), crop_b.strip().upper()]
    
    df = crop_df[
        (crop_df['State_Name'] == region_upper) &
        (crop_df['Crop_Year'].between(start_year, latest_year)) &
        (crop_df['Crop'].isin(crops_upper))
    ]
    
    if df.empty or len(df['Crop'].unique()) < 2:
        return json.dumps({"error": f"Not enough comparative data found for '{crop_a}' and '{crop_b}' in '{region}' for the last {years} years."})

    analysis = df.groupby('Crop').agg(
        average_yield_tonnes_per_hectare=('Yield', 'mean'),
        total_production_tonnes=('Production', 'sum'),
        years_of_data=('Crop_Year', 'nunique')
    ).reset_index()

    rainfall = rain_df[rain_df['State_Name'] == region_upper]['ANNUAL'].mean()

    result = {
        "region": region,
        "comparison_period": f"{start_year}-{latest_year} ({years} years)",
        "crop_comparison_metrics": analysis.to_dict('records'),
        "rainfall_context_mm": round(rainfall, 2) if pd.notna(rainfall) else "N/A",
        "data_sources_used": DATA_SOURCES
    }
    
    return json.dumps(result, indent=2)

