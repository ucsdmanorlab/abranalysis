import streamlit as st
from config import AppConfig

def create_data_tab():
    """Create the data upload and configuration tab."""
    tab1 = st.sidebar.tabs(["Data", "Plotting and Analysis"])[0]
    
    uploaded_files = tab1.file_uploader(
        "**Upload files to analyze:**", 
        type=AppConfig.SUPPORTED_FILE_TYPES, 
        accept_multiple_files=True
    )
    
    inputs = tab1.expander("Input data properties", expanded=True)
    
    return {
        'uploaded_files': uploaded_files,
        'units': inputs.selectbox("Units used in collection", options=AppConfig.UNITS, index=0),
        'is_click': inputs.radio("Tone or click? (for .arf files)", AppConfig.TONE_CLICK_OPTIONS, horizontal=True) == "Click",
        'is_atten': inputs.toggle("dB saved as attenuation", value=False),
        'inputs_container': inputs
    }

def create_plotting_tab(distinct_freqs, distinct_dbs, level):
    """Create the plotting and analysis tab."""
    tab2 = st.sidebar.tabs(["Data", "Plotting and Analysis"])[1]
    
    # Output settings
    outputs = tab2.expander("Output and plot settings", expanded=False)
    return_units = outputs.selectbox("Units for plots and outputs", options=AppConfig.UNITS, index=0)
    
    # Auto-configure Y limits based on units
    y_defaults = get_y_defaults(return_units)
    auto_y = outputs.toggle("Auto Y-axis scaling", value=True)
    
    return {
        'return_units': return_units,
        'auto_y': auto_y,
        'y_min': outputs.number_input("Y-axis minimum", value=y_defaults['min'], disabled=auto_y),
        'y_max': outputs.number_input("Y-axis maximum", value=y_defaults['max'], disabled=auto_y),
        'show_legend': outputs.toggle("Show legend", True),
        'show_peaks': outputs.toggle("Show peaks", True),
        'serif_font': outputs.toggle("Use serif fonts", value=False),
        # ... other settings
    }

def get_y_defaults(return_units):
    """Get default Y-axis limits based on units."""
    if return_units == 'Nanovolts':
        return {'min': AppConfig.DEFAULT_Y_MIN_NANOVOLTS, 'max': AppConfig.DEFAULT_Y_MAX_NANOVOLTS}
    return {'min': AppConfig.DEFAULT_Y_MIN_MICROVOLTS, 'max': AppConfig.DEFAULT_Y_MAX_MICROVOLTS}