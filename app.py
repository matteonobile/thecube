import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from io import StringIO

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Factor Exposure Cube")

# --- Helper Functions ---

def load_data(uploaded_file=None):
    """
    Loads data from the CSV file or a hardcoded fallback if file is missing.
    Specific logic to parse rows 7-12 of the provided format.
    """
    if uploaded_file is not None:
        # Read the file if uploaded by user in the sidebar
        df_raw = pd.read_csv(uploaded_file, header=None)
    elif os.path.exists('Factor Exposures.xlsx - Sheet1.csv'):
        # Read the local file if it exists
        df_raw = pd.read_csv('Factor Exposures.xlsx - Sheet1.csv', header=None)
    else:
        # Fallback data (based on your provided file) for robustness
        data = """
Name,Growth,Quality,Size
GEC,0.26,0.175,0.51
Global,0.035,0.0175,0.31
Growth Idx,0.49,0.375,0.47
Small Idx,-0.35,-0.2675,-1.46
Quality (etf),0.3,0.3975,0.46
"""
        df_raw = pd.read_csv(StringIO(data))
        # Adjust format to match the complex CSV structure logic below
        return df_raw 

    # --- Parsing Logic for the specific CSV structure ---
    # The snippet showed the block starts around row 7 (index 6).
    # We look for the row containing "GEC" and "Global"
    
    # Slice the specific block (Rows 7-12 approx, based on inspection)
    # We extract columns: 0 (Name), 1 (Growth), 3 (Quality), 7 (Size)
    # Note: Adjust indices based on actual file inspection
    
    target_data = []
    
    # Iterate to find the block
    start_row = -1
    for i, row in df_raw.iterrows():
        # Look for the header row of this block
        if row[1] == 'Growth' and row[7] == 'Size':
            start_row = i + 1
            break
            
    if start_row == -1:
        # If header not found, assume standard dataframe format from fallback
        if 'Growth' in df_raw.columns:
            return df_raw
        else:
            st.error("Could not locate data block in CSV.")
            return pd.DataFrame()

    # Extract the 5 lines of data following the header
    block = df_raw.iloc[start_row:start_row+5]
    
    # specific columns based on file inspection: 
    # Col 0: Name, Col 1: Growth, Col 3: Quality, Col 7: Size
    clean_df = pd.DataFrame({
        'Name': block[0],
        'Growth': pd.to_numeric(block[1], errors='coerce'),
        'Quality': pd.to_numeric(block[3], errors='coerce'),
        'Size': pd.to_numeric(block[7], errors='coerce')
    })
    
    return clean_df

def round_up_to_half(value):
    """Rounds a value up to the nearest 0.5 increment."""
    return np.ceil(abs(value) / 0.5) * 0.5

def main():
    st.title("🧊 Factor Exposure Visualization")
    st.markdown("""
    **3D Analysis of Portfolio Factors.** The **Global (Benchmark)** is centered at (0,0,0). All other positions are relative to the benchmark.
    """)

    # --- Data Loading ---
    # Optional: Allow file upload
    uploaded_file = st.sidebar.file_uploader("Upload Factor CSV", type=['csv'])
    
    df = load_data(uploaded_file)
    
    if df.empty:
        st.stop()

    # --- Data Processing ---
    # 1. Identify Global (Benchmark)
    try:
        global_row = df[df['Name'] == 'Global'].iloc[0]
    except IndexError:
        st.error("Row 'Global' not found in data.")
        st.stop()
        
    # 2. Center data on Global (Relative Calculation)
    df['Growth_Rel'] = df['Growth'] - global_row['Growth']
    df['Quality_Rel'] = df['Quality'] - global_row['Quality']
    df['Size_Rel'] = df['Size'] - global_row['Size']
    
    # 3. Calculate Limits (Cube Dimensions)
    # Find max absolute deviation for each axis
    max_g = df['Growth_Rel'].abs().max()
    max_q = df['Quality_Rel'].abs().max()
    max_s = df['Size_Rel'].abs().max()
    
    # Round up to nearest 0.5
    lim_g = round_up_to_half(max_g)
    lim_q = round_up_to_half(max_q)
    lim_s = round_up_to_half(max_s)
    
    # For a symmetrical "Cube" feel, we might want to use the largest limit for all,
    # or keep it a rectangular box. The user said "build the cube around those positions".
    # We will use individual limits to fit the data tightly but cleanly.
    
    # --- Plotting ---
    fig = go.Figure()

    # Define common marker styles
    for i, row in df.iterrows():
        name = row['Name']
        x, y, z = row['Growth_Rel'], row['Quality_Rel'], row['Size_Rel']
        
        # Style logic
        if name == 'GEC':
            color = '#EF553B'  # Red/Orange for Portfolio
            size = 12
            opacity = 1.0
            symbol = 'diamond'
            # Add projections for GEC only
            # Floor (Growth vs Quality)
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[-lim_s],
                mode='markers', marker=dict(color='gray', size=6, opacity=0.3),
                name=f"{name} Shadow (Floor)", showlegend=False, hoverinfo='skip'
            ))
            # Wall (Growth vs Size)
            fig.add_trace(go.Scatter3d(
                x=[x], y=[-lim_q], z=[z],
                mode='markers', marker=dict(color='gray', size=6, opacity=0.3),
                name=f"{name} Shadow (Wall)", showlegend=False, hoverinfo='skip'
            ))
             # Back (Quality vs Size)
            fig.add_trace(go.Scatter3d(
                x=[-lim_g], y=[y], z=[z],
                mode='markers', marker=dict(color='gray', size=6, opacity=0.3),
                name=f"{name} Shadow (Back)", showlegend=False, hoverinfo='skip'
            ))
            
            # Add "Drop lines" to shadows to make position super clear
            # Line to floor
            fig.add_trace(go.Scatter3d(
                x=[x, x], y=[y, y], z=[z, -lim_s],
                mode='lines', line=dict(color='gray', width=2, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ))
            
        elif name == 'Global':
            color = 'black'
            size = 10
            opacity = 1.0
            symbol = 'circle'
        else:
            # Other portfolios
            color = '#636EFA' # Blue
            size = 8
            opacity = 0.5
            symbol = 'circle'

        # Plot the Point
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=size, color=color, opacity=opacity, symbol=symbol),
            text=[name], textposition="top center",
            name=name,
            hovertemplate=
            f"<b>{name}</b><br>" +
            f"Growth (Rel): %{{x:.2f}}<br>" +
            f"Quality (Rel): %{{y:.2f}}<br>" +
            f"Size (Rel): %{{z:.2f}}<extra></extra>"
        ))

    # --- Draw the Transparent "Room" (Bounding Box) ---
    # We draw the mesh of the limits
    x_corn = [-lim_g, -lim_g, lim_g, lim_g, -lim_g, -lim_g, lim_g, lim_g]
    y_corn = [-lim_q, lim_q, lim_q, -lim_q, -lim_q, lim_q, lim_q, -lim_q]
    z_corn = [-lim_s, -lim_s, -lim_s, -lim_s, lim_s, lim_s, lim_s, lim_s]
    
    fig.add_trace(go.Mesh3d(
        x=x_corn, y=y_corn, z=z_corn,
        color='lightgray',
        opacity=0.05, # Very faint
        alphahull=0,
        name='Bounds'
    ))

    # --- Layout Update ---
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Growth (Rel)', range=[-lim_g, lim_g], backgroundcolor="white"),
            yaxis=dict(title='Quality (Rel)', range=[-lim_q, lim_q], backgroundcolor="white"),
            zaxis=dict(title='Size (Rel)', range=[-lim_s, lim_s], backgroundcolor="white"),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1) # Force a cube look
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        height=700,
        legend=dict(x=0, y=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Display the calculated data for verification
    with st.expander("See processed data"):
        st.dataframe(df[['Name', 'Growth', 'Growth_Rel', 'Quality', 'Quality_Rel', 'Size', 'Size_Rel']])

if __name__ == "__main__":
    main()