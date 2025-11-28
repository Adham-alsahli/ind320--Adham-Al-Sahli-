import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from pymongo.server_api import ServerApi
from pathlib import Path
import tomllib
from scipy.fftpack import dct, idct
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px
import folium
from streamlit_folium import st_folium
import json
import time 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
import statsmodels.api as sm


# Robust import av STL
HAS_STATSMODELS = False
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
except ModuleNotFoundError:
    #  håndterer dette senere i stl_decomposition()
    pass

from scipy.signal import spectrogram
import requests as rquests


try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        print("This app must be started with: streamlit run Apps/App.py\nExiting to avoid missing Streamlit runtime.")
        raise SystemExit(0)
except Exception:
    # If import fails or no context, instruct and exit when not run by streamlit.
    print("This app should be run with 'streamlit run' for full functionality. Exiting.")
    raise SystemExit(0)

st.set_page_config(page_title="IND320 App", layout="wide")


@st.cache_data
def load_data():
    # Laster data fra CSV-fil (bruker en robust sti relativt til app-mappen)
    base = Path(__file__).resolve().parent.parent  # project root (Ind320)
    csv_path = base / "open-meteo-subset.csv"
    # also accept Data/ subfolder for backward compatibility
    if not csv_path.exists():
        alt = base / "Data" / "open-meteo-subset.csv"
        if alt.exists():
            csv_path = alt
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}. Place 'open-meteo-subset.csv' in the project root or Data/ folder.")
    return pd.read_csv(csv_path)





# Prøv å laste data
try:
    df = load_data()
except FileNotFoundError:
    df = None

st.sidebar.title("Navigasjon 🧭")

page = st.sidebar.radio("Velg side:",
                            ["Hjem 🏠",
                             "Data-tabell 📋",
                             "Weather-Plot 🌦️",
                             "SnowDrift ❄️",
                             "SPC and LOF Analyse📈",
                             "MongoDB",
                             "STL Og Spectrogram 📊",
                             "Map 🗺️",
                             "Sliding Window Correlation 📉",
                             "Forecasting 📅"
                            ])





# hjemmeside
if page == "Hjem 🏠":
    st.title("IND320")
    st.markdown("""
    ## Velkommen til Min App!
                
    Dette dashbordet er laget for å gi innsikt i energiproduksjon og meteorologiske data. \n
    Bruk sidemenyen for å navigere mellom de ulike seksjonene og utforske analyser og visualiseringer.
   
    
    """)

elif page == "MongoDB":
    st.title("MongoDB Data Visualization")

    #  Mongo tilkobling
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi

    @st.cache_resource # Cache for å unngå reconnect hver gang
    def load_data_from_mongo():
        uri = st.secrets["MongoDB"]["uri"]
        database = st.secrets["MongoDB"]["database"]
        
        client = MongoClient(uri, server_api=ServerApi("1"))
        db = client[database]
        
        #loada production data
        prod_data = list(db["production_data"].find())
        prod_data_df = pd.DataFrame(prod_data)

        #Loada consumption data
        cons_data = list(db["consumption_data"].find())
        cons_data_df = pd.DataFrame(cons_data)

        # konvertere timestamp 
        for df in [prod_data_df, cons_data_df]:
            if 'starttime' in df.columns:
                df['starttime'] = pd.to_datetime(df['starttime'])

        return prod_data_df, cons_data_df
    
        
    #Layout: to kolonner 
    left_column, right_column = st.columns(2)

    #starter session state for selection
    if "selected_area" not in st.session_state:
        st.session_state["selected_area"] = "NO1"
    
    if "selected_group" not in st.session_state:
        st.session_state["selected_group"] = ["hydro", "wind", "solar", "thermal", "other"]

    #sjekker hvis mongo data er lastet
    if "mongo_data" not in st.session_state:
        prod_data_df, cons_data_df = load_data_from_mongo()
        st.session_state["mongo_data"] = prod_data_df, cons_data_df
        st.write("Data lastet fra MongoDB.")
    else:
        prod_data_df, cons_data_df = st.session_state["mongo_data"]
        st.write("Bruker cachet data fra MongoDB.")

    
    # VENSTRE: Radio + Pie 
    with left_column:
        st.subheader("Production Share by Group")
        price_areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]
        selected_area = st.radio("Select Price Area", 
                                price_areas, 
                                key="area_radio",
                                index=price_areas.index(st.session_state["selected_area"]))
           
    #update session state when selection changes 
    if selected_area != st.session_state["selected_area"]:
        st.session_state["selected_area"] = selected_area

    #filter data for selected area
    area_filtered_prod_data = prod_data_df[prod_data_df["pricearea"] == selected_area]
    prorduction_by_group = area_filtered_prod_data.groupby("productiongroup")["quantitykwh"].sum().sort_values(ascending=False)

    #lager pie chart 
    fig_pie =px.pie(
        names = prorduction_by_group.index,
        values = prorduction_by_group.values,
        title = f"Production share by Group in {selected_area} (2021)",)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)
    

    # HØYRE: Pills/Multiselect + Måned + Line
    with right_column:
        st.subheader("Monthly Production Trend")

        #piller for selecting production groups
        all_groups = ["hydro", "wind", "solar", "thermal", "other"]
        selected_group = st.pills("Select Production Group", 
                                 all_groups,
                                 selection_mode="multi",
                                 default=st.session_state["selected_group"],
                                 key="group_pills")
    #update session state when selection changes
    if set(selected_group) != set(st.session_state["selected_group"]):
        st.session_state["selected_group"] = selected_group

    #selectbox for month selection
    months = list(range(1,13))
    selected_month = st.selectbox("Select Month", months, index=0)
    #filtere data basert på valget gruppe og måned
    filtered_data = prod_data_df[
        (prod_data_df["starttime"].dt.year == 2021) &
        (prod_data_df["starttime"].dt.month == selected_month) &
        (prod_data_df["productiongroup"].isin(selected_group)) &
        (prod_data_df["pricearea"] == selected_area)
    ]

    #lager piovt table 
    pivot_table = filtered_data.pivot_table(values="quantitykwh",
                                            index=["starttime"],
                                            columns=["productiongroup"],
                                            aggfunc="sum",)
    #bruk ploty for line chart
    

    #konverter pivot table til lang format for plotly
    pivot_table_reset = pivot_table.reset_index().melt(id_vars="starttime",
                                                       value_name="quantitykwh",
                                                       var_name="productiongroup",
                                                       )
    fig_line = px.line(
        pivot_table_reset,
        x="starttime",
        y="quantitykwh",
        color="productiongroup",
        title=f"Hourly Production by Group in {selected_area} - Month {selected_month} 2021",)
    
    fig_line.update_layout(xaxis_title="Time",
                            yaxis_title="Production (kWh)",
                            legend_title="Production Group",)
    st.plotly_chart(fig_line, use_container_width=True)
    
    
        # Expander 
    with st.expander("Data Source"):
        st.markdown("""
Dataene som vises på denne siden kommer fra **Elhub sitt Energy Data API** (https://api.elhub.no/),  
som gir timesdata for produksjon i ulike energigrupper på tvers av norske prisområder.  

For mer informasjon, se [Elhub API Services](https://api.elhub.no/).

        """)

# STL and Spectrogram
elif page == "STL Og Spectrogram 📊":
    st.title("New Page A: STL and Spectrogram")

    if "selected_group" not in st.session_state:
        st.session_state["selected_group"] = ["hydro", "wind", "solar", "thermal", "other"]
    if "selected_area" not in st.session_state:
        st.session_state["selected_area"] = "NO1"
# Tabs for STL and Spectrogram
    tabel1 , tabel2 = st.tabs(["STL Analysis", "Spectrogram Analysis"])

    selected_area = st.session_state.get("selected_area", "NO1")
    selected_groups = st.session_state.get("selected_group", ["hydro", "wind", "solar", "thermal", "other"])
# data loading function
    @st.cache_data(ttl= 6000)
    def load_mongo_data():
    # Les fra Streamlit Secrets (Cloud UI)
        uri = st.secrets["MongoDB"]["uri"]
        database = st.secrets["MongoDB"]["database"]
        
        client = MongoClient(uri, server_api=ServerApi("1"))
        db = client[database]
        
        #loada production data
        prod_data = list(db["production_data"].find())
        prod_data_df = pd.DataFrame(prod_data)

        #Loada consumption data
        cons_data = list(db["consumption_data"].find())
        cons_data_df = pd.DataFrame(cons_data)

        #konvertere timestamp
        for df in [prod_data_df, cons_data_df]:
            if 'starttime' in df.columns:
                df['starttime'] = pd.to_datetime(df['starttime'])
        return prod_data_df
    
    if selected_area not in st.session_state:
        st.session_state["selected_area"] = "NO1"

    if selected_groups not in st.session_state:
        st.session_state["selected_group"] = ["hydro", "wind", "solar", "thermal", "other"]

    #sjekk om mongo data er lastet
    if "mongo_data" not in st.session_state:
        prod_data_df, cons_data_df = load_mongo_data()
        st.session_state["mongo_data"] = prod_data_df, cons_data_df
        st.write("Data lastet fra MongoDB.")

    else:
        prod_data_df, cons_data_df = st.session_state["mongo_data"]
        st.write("Bruker cachet data fra MongoDB.")
    

#stL and spectrogram code functions

    def stl_decomposition_plotly(df, price_area="NO5", production_group="solar",
                                 period=24, seasonal=7, trend=None, robust=True):

        if not HAS_STATSMODELS:
            st.error("statsmodels mangler – installer det for å bruke STL")
            return None

        data = df[(df["pricearea"] == price_area) &
                  (df["productiongroup"] == production_group)]

        ts = data["quantitykwh"]
        ts.index = pd.to_datetime(data["starttime"])
        ts.sort_index(inplace=True)

        # STL
        stl = STL(ts, period=period, seasonal=seasonal,
                  trend=trend if trend else period+1, robust=robust)
        result = stl.fit()

        # Plotly figur
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=["Original", "Trend", "Seasonal", "Residual"]
        )

        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Original"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts.index, y=result.trend, name="Trend"), row=2, col=1)
        fig.add_trace(go.Scatter(x=ts.index, y=result.seasonal, name="Seasonal"), row=3, col=1)
        fig.add_trace(go.Scatter(x=ts.index, y=result.resid, name="Residual"), row=4, col=1)

        fig.update_layout(height=900, showlegend=False)
        return fig


    # ============================================================
    # 2) SPEKTROGRAM MED PLOTLY
    # ============================================================

    def create_spectrogram_plotly(df, price_area="NO5",
                                  production_group="solar",
                                  window_length=400, overlap=150):

        data = df[(df["pricearea"] == price_area) &
                  (df["productiongroup"] == production_group)]

        ts = data["quantitykwh"]
        ts.index = pd.to_datetime(data["starttime"])
        ts.sort_index(inplace=True)

        # SciPy spectrogram
        f, t, Sxx = spectrogram(ts, fs=1, nperseg=window_length,
                                noverlap=overlap)

        # Konverter til dB
        Z = 10 * np.log10(Sxx + 1e-12)

        fig = go.Figure(data=go.Heatmap(
            z=Z,
            x=t,
            y=f,
            colorscale='Plasma',
            colorbar=dict(title="Power (dB)")
        ))

        fig.update_layout(
            title=f"Spectrogram of {production_group} ({price_area})",
            xaxis_title="Time (hours)",
            yaxis_title="Frequency (cycles/hour)",
            height=700
        )

        return fig


    elhub_dataBase = load_mongo_data()

    with tabel1:
        st.header("STL Decomposition")
        selected_group_stl = st.radio("Select Production Group", st.session_state["selected_group"])
        fig = stl_decomposition_plotly(elhub_dataBase,
                                       price_area=selected_area,
                                       production_group=selected_group_stl)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tabel2:
        st.header("Spectrogram Analysis")
        selected_group_spec = st.radio("Select Production Group (Spectrogram)",
                                       st.session_state["selected_group"],
                                       key="spec")
        fig = create_spectrogram_plotly(elhub_dataBase,
                                        price_area=selected_area,
                                        production_group=selected_group_spec)
        st.plotly_chart(fig, use_container_width=True)


# DATA-TABELL
elif page == "Data-tabell 📋":
    st.header("Data-tabell")
    
    if "selected_area" not in st.session_state:
        st.session_state["selected_area"] = "NO1"
    
    if "selected_group" not in st.session_state:
        st.session_state["selected_group"] = ["hydro", "wind", "solar", "thermal", "other"]

    # price areas to coordinates 
    area_coordinates = {
        "NO1": (59.91, 10.75),  # Oslo
    "NO2": (58.15, 7.99),   # Kristiansand
    "NO3": (63.43, 10.39),  # Trondheim
    "NO4": (69.65, 18.96),  # Tromsø
    "NO5": (60.39, 5.32)    # Bergen
    }
    selected_area = st.session_state.get("selected_area", "NO1")
    selected_groups = st.session_state.get("selected_group", ["hydro", "wind", "solar", "thermal", "other"])

    # current selection display
    st.info(f"Valgt prisområde: **{selected_area}** | Valgte produksjonsgrupper: **{', '.join(selected_groups)}**")

    lat, lon = area_coordinates.get(selected_area)
    selected_year =2021

    @st.cache_data
    def load_data_from_api(lat, lon, year, variables=["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]):
        url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={year}-01-01&end_date={year}-12-31&hourly="
        for var in variables:
            url += f"{var}," if var != variables[-1] else f"{var}"
        url += "&timezone=Europe%2FOslo"

        print(f"Fetching data from URL: {url}")
        response = rquests.get(url)
        if response.status_code == 200:
            data = response.json()
        
            hourly_data = data.get("hourly", {})
            df_api = pd.DataFrame(hourly_data)

            # time to column 
            df_api["time"] = pd.to_datetime(df_api["time"], errors="coerce")

            return df_api
        else:
            st.error(f"Feil ved henting av data fra API: {response.status_code}")
            return None
        
    if "weather_data" not in st.session_state:
        st.session_state["weather_data"] = load_data_from_api(lat, lon, selected_year,variables=["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"])
    data = st.session_state["weather_data"]   
    first_M = data[data["time"].dt.month == 1]

    #Row wise table: one row per orginial column (except "time")

    table_ROWWISE = pd.DataFrame({
        "variable": [col for col in first_M.columns if col != "time"],
        "january_values": [first_M[col].tolist() for col in first_M.columns if col != "time"]
    })
    # show table with line chart 
    st.data_editor(
        table_ROWWISE,
        column_config={
            "january_values": st.column_config.LineChartColumn(
                label= f"Values for January 2021 at {selected_area}",
                width="large")                
            
        },
    )
# PLOT 
elif page == "Weather-Plot 🌦️":
    st.header("Visualisering av data")
    if df is None:
        st.error("Fant ikke datafilen 'open-meteo-subset.csv'.")
    else:
        # Fast forbehandling (ingen valg i UI)
        work = df.copy()
        work["time"] = pd.to_datetime(work["time"], errors="coerce")
        work["month"] = work["time"].dt.to_period("M").astype(str)

        # Finn vindretningskolonne og unwrap ALLTID
        dir_candidates = [c for c in work.columns if "wind_direction" in c.lower()]
        direction_col = dir_candidates[0] if dir_candidates else None
        if direction_col:
            rad = np.deg2rad(work[direction_col].to_numpy())
            work[direction_col] = np.rad2deg(np.unwrap(rad))

        # Alltid glatting (rolling mean) med fast vindu=5
        numeric_cols = [c for c in work.columns if c not in ("time", "month")]
        work[numeric_cols] = work[numeric_cols].rolling(window=5, min_periods=1).mean()

        # Ui kun for kolonnevalg og månedsspenn
        col_choice = st.selectbox("Velg kolonne:", ["Alle"] + numeric_cols)
        #
        months = sorted(work["month"].unique())
        month_range = st.select_slider(
            "Velg måned(er):",
            options=months,
            value=(months[0], months[0])
        )

        if isinstance(month_range, tuple):
            subset = work[(work["month"] >= month_range[0]) & (work["month"] <= month_range[1])].copy()
        else:
            subset = work[work["month"] == month_range].copy()

        st.subheader(f"Plot for {col_choice} ({month_range})")

        # Hjelpefunksjon: fast Z-score (for Alle)
        def zscore(dfX):
            Z = (dfX - dfX.mean()) / dfX.std(ddof=0)
            return Z.replace([np.inf, -np.inf], np.nan).fillna(0)
        #
        if col_choice == "Alle":
            plot_df = subset.set_index("time")[numeric_cols].copy()
            plot_df = zscore(plot_df)  # ← ALLTID z-score-normalisert
            st.line_chart(plot_df, use_container_width=True)
           
        else:
            single_df = subset.set_index("time")[[col_choice]]
            st.line_chart(single_df, use_container_width=True)
           

# New_B-side: Outlier/SPC and Anomaly/LOF
elif page == "SPC and LOF Analyse📈":
    st.header("SPC and LOF Analyse📈 Page")
#
    selected_area = st.session_state.get("selected_area", "NO1")

    #price areas to coordinates map
    area_coordinates = {
        "NO1": (59.91, 10.75),  # Oslo
        "NO2": (58.15, 7.99),   # Kristiansand
        "NO3": (63.43, 10.39),  # Trondheim
        "NO4": (69.65, 18.96),  # Tromsø
        "NO5": (60.39, 5.32)    # Bergen
    }
    lat, lon = area_coordinates.get(selected_area)
    selected_year =2021

    tabel1 , tabel2 = st.tabs(["Outlier/SPC Analysis", "Anomaly/LOF Analysis"])

    #data loading function
    @st.cache_data(ttl= 6000)
    def load_data_from_api(lat, lon, year, variables=["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]):
        url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={year}-01-01&end_date={year}-12-31&hourly="
        for var in variables:
            url += f"{var}," if var != variables[-1] else f"{var}"
        url += "&timezone=Europe%2FOslo"

        print(f"Fetching data from URL: {url}")
        response = rquests.get(url)
        if response.status_code == 200:
            data = response.json()
        
            hourly_data = data.get("hourly", {})
            df_api = pd.DataFrame(hourly_data)

            # time to column 
            df_api["time"] = pd.to_datetime(df_api["time"], errors="coerce")

            return df_api
        else:
            st.error(f"Feil ved henting av data fra API: {response.status_code}")
            return None
# Functions for Outlier/SPC and Anomaly/LOF analysis
    def high_pass_filter(data, cutoff_frequency):
        # Perform DCT
        data_transformed = dct(data, norm='ortho')
        
        # Zero out low frequency components
        filtered_transformed = np.copy(data_transformed)
        filtered_transformed[:cutoff_frequency] = 0    
        # Perform inverse DCT
        filtered_data = idct(filtered_transformed, norm='ortho')
        return filtered_data
    
    def detect_temperature_outliers(time, temperature, freq_cutoff=100, num_std=3):
    # High-pass filter
        satv = high_pass_filter(temperature, freq_cutoff)

        # Robust stats
        median = np.median(satv)
        mad = np.median(np.abs(satv - median))
        threshold = 1.4826 * mad

        upper_control_limit = median + num_std * threshold
        lower_control_limit = median - num_std * threshold

        upper_curve = temperature + (upper_control_limit - satv)
        lower_curve = temperature + (lower_control_limit - satv)

        outliers = (satv > upper_control_limit) | (satv < lower_control_limit)
        outlier_idx = np.where(outliers)[0]

        # --- PLOTTING WITH PLOTLY ---
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time, y=temperature,
            mode='lines',
            name='Temperature',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=time[outliers], y=temperature[outliers],
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=8)
        ))

        fig.add_trace(go.Scatter(
            x=time, y=upper_curve,
            mode='lines',
            name='Upper Control Limit',
            line=dict(color='green', dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=time, y=lower_curve,
            mode='lines',
            name='Lower Control Limit',
            line=dict(color='green', dash='dash')
        ))

        fig.update_layout(
            title="Temperature Outliers (SPC)",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            height=500,
            legend=dict(orientation='h')
        )

        summary = {
            "outlier_indices": outlier_idx,
            "outlier_times": time[outliers],
            "outlier_temperatures": temperature[outliers],
            "num_outliers": len(outlier_idx),
            "upper_control_limit": upper_control_limit,
            "lower_control_limit": lower_control_limit,
            "Robust_median": median,
            "robust_std": threshold
        }

        return fig, summary


    def detect_lof_outliers(time, data_variable, n_neighbors=20, contamination=0.01):
        # reshape for LOF
        D = np.array(data_variable).reshape(-1, 1)

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        labels = lof.fit_predict(D)
        scores = -lof.negative_outlier_factor_

        outlier_mask = labels == -1
        outlier_idx = np.where(outlier_mask)[0]

        # --- PLOTLY ---
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time, y=data_variable,
            mode="lines",
            name="Selected Variable",
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=time[outlier_mask], y=data_variable[outlier_mask],
            mode="markers",
            name="Outliers",
            marker=dict(color='red', size=8)
        ))

        fig.update_layout(
            title="LOF Anomaly Detection",
            xaxis_title="Time",
            yaxis_title="Value",
            height=500,
            legend=dict(orientation="h")
        )

        summary = {
            "outlier_indices": outlier_idx,
            "outlier_times": time[outlier_mask],
            "outlier_selected_variable": data_variable[outlier_mask],
            "num_outliers": len(outlier_idx),
            "lof_scores": scores[outlier_mask]
        }

        return fig, summary

    if "weather_data" in st.session_state:
        weather_data = st.session_state["weather_data"]
        st.write("Data loaded from API.")

    else:
        weather_data = load_data_from_api(lat, lon, selected_year,variables=["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"])
        st.session_state["weather_data"] = weather_data
        st.write("Data loaded from API.")


    with tabel1:
        st.subheader("Outlier/SPC Analysis on Temperature Data")
        freq_cutoff = st.slider("Select Frequency Cutoff:", 10, 500, 100)
        num_std = st.slider("Std deviations:", 1, 5, 3)

        fig, summary = detect_temperature_outliers(
            weather_data["time"],
            weather_data["temperature_2m"].values,
            freq_cutoff,
            num_std
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabel2:
        st.subheader("LOF Anomaly Detection")
        selected_area = st.radio("Select Variable", ["precipitation", "wind_speed_10m", "wind_gusts_10m"])
        contamination = st.slider("Contamination:", 0.001, 0.1, 0.01)
        n_neighbors = st.slider("Neighbors:", 5, 50, 20)

        fig, summary = detect_lof_outliers(
            weather_data["time"].values,
            weather_data[selected_area].values,
            n_neighbors,
            contamination
        )
        st.plotly_chart(fig, use_container_width=True)


elif page == "Map 🗺️":
    st.header("Map 🗺️ Page")

    @st.cache_data(ttl= 6000)
    def load_data_from_mongo():
        uri = st.secrets["MongoDB"]["uri"]
        database = st.secrets["MongoDB"]["database"]
        
        client = MongoClient(uri, server_api=ServerApi("1"))
        db = client[database]
        
        #loada production data
        prod_data = list(db["production_data"].find())
        prod_data_df = pd.DataFrame(prod_data)

        #Loada consumption data
        cons_data = list(db["consumption_data"].find())
        cons_data_df = pd.DataFrame(cons_data)

        # konvertere timestamp 
        for df in [prod_data_df, cons_data_df]:
            if 'starttime' in df.columns:
                df['starttime'] = pd.to_datetime(df['starttime'])

        return prod_data_df, cons_data_df
    
    with open("Data/file.geojson", "r", encoding="utf-8") as f:
        geo_data = json.load(f)


        for area in geo_data["features"]:
            area["properties"]["ElSpotOmr"] = area["properties"]["ElSpotOmr"].replace(" ", "")
            
    #sjekker hvis mongo data er lastet
    if "mongo_data" not in st.session_state:
        prod_data_df, cons_data_df = load_data_from_mongo()
        st.session_state["mongo_data"] = prod_data_df, cons_data_df
        st.write("Data lastet fra MongoDB.")
    else:
        prod_data_df, cons_data_df = st.session_state["mongo_data"]
        st.write("Bruker cachet data fra MongoDB.")

    col1 , col2 = st.columns([1,1])

    with col1:
        group = st.selectbox("Select Energy Group", ["hydro", "wind", "solar", "thermal", "other"])
    
    with col2:
        start_date, end_date = st.date_input("Select time range", value=(cons_data_df["starttime"].min(), cons_data_df["starttime"].max()))
    
    col3 , col4 = st.columns([1,1])

    with col3:
        area_type = st.radio("Selected Area",["NO1", "NO2", "NO3", "NO4", "NO5"], index=0,horizontal=True)

    with col4:
        group_type = st.radio("Select Group Type", ["production", "consumption"], index=0, horizontal=True)

    if group_type == "production":
        data_filtered = prod_data_df[(prod_data_df["starttime"] >= pd.to_datetime(start_date)) & (prod_data_df["starttime"] <= pd.to_datetime(end_date))]

    elif group_type == "consumption":
        data_filtered = cons_data_df[(cons_data_df["starttime"] >= pd.to_datetime(start_date)) & (cons_data_df["starttime"] <= pd.to_datetime(end_date))]


    data_area_grouped =(
        data_filtered.groupby("pricearea")["quantitykwh"]
        .mean().reset_index(name ="avg_value")
    )

    # sentral posisjon for norge
    map_center = folium.Map(location=[65, 15], zoom_start=5, tiles ="CartoDB positron")

    # legg til chrolopleth lag for gjennomsnittlig verdier
    choropleth = folium.Choropleth(
        geo_data=geo_data,
        name="choropleth",
        data=data_area_grouped,
        columns=["pricearea", "avg_value"],
        key_on="feature.properties.ElSpotOmr",
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name= "Mean Value",).add_to(map_center)
    
    #marker valgete area
    folium.GeoJson(
        geo_data,
        name="Selected Area",
        style_function=lambda feature: {
            "fillColor": "yellow" if feature["properties"]["ElSpotOmr"] == area_type else "black",
            "color": "yellow" if feature["properties"]["ElSpotOmr"] == area_type else 1,
            "weight": 3,
            "fillOpacity": 0
        },
    ).add_to(map_center)


    # vis kartet 

    Map_display = st_folium(map_center, width=None, height=800)


    #vis valgte data
    if Map_display and "last_clicked" in Map_display and Map_display["last_clicked"]:
        lat = Map_display["last_clicked"]["lat"]
        lon = Map_display["last_clicked"]["lng"]
        st.session_state["selected_location"] = (lat, lon)


    #hent data
        hent_data =f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        try:
            hent_data_response = rquests.get(hent_data)
            hent_data_response.raise_for_status()  # Raise an error for bad responses

            if "elevation" in hent_data_response.json():
                elevation = hent_data_response.json()["elevation"]
                st.success(f"Elevation at selected location: {elevation} meters")
            

        except Exception as e:
            st.error(f"Error fetching elevation data: {e}")


elif page == "SnowDrift ❄️":
    st.header("SnowDrift ❄️ ")


    def compute_Qupot(hourly_wind_speeds, dt=3600):
        """
        Compute the potential wind-driven snow transport (Qupot) [kg/m]
        by summing hourly contributions using u^3.8.
        
        Formula:
        Qupot = sum((u^3.8) * dt) / 233847
        """
        total = sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847
        return total

    def sector_index(direction):
        """
        Given a wind direction in degrees, returns the index (0-15)
        corresponding to a 16-sector division.
        """
        # Center the bin by adding 11.25° then modulo 360 and divide by 22.5°
        return int(((direction + 11.25) % 360) // 22.5)

    def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
        """
        Compute the cumulative transport for each of 16 wind sectors.
        
        Parameters:
        hourly_wind_speeds: list of wind speeds [m/s]
        hourly_wind_dirs: list of wind directions [degrees]
        dt: time step in seconds
        
        Returns:
        A list of 16 transport values (kg/m) corresponding to the sectors.
        """
        sectors = [0.0] * 16
        for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
            idx = sector_index(d)
            sectors[idx] += ((u ** 3.8) * dt) / 233847
        return sectors

    def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
        """
        Compute various components of the snow drifting transport according to Tabler (2003).
        
        Parameters:
        T: Maximum transport distance (m)
        F: Fetch distance (m)
        theta: Relocation coefficient
        Swe: Total snowfall water equivalent (mm)
        hourly_wind_speeds: list of wind speeds [m/s]
        dt: time step in seconds
        
        Returns:
        A dictionary containing:
            Qupot (kg/m): Potential wind-driven transport.
            Qspot (kg/m): Snowfall-limited transport.
            Srwe (mm): Relocated water equivalent.
            Qinf (kg/m): The controlling transport value.
            Qt (kg/m): Mean annual snow transport.
            Control: Process controlling the transport (wind or snowfall).
        """
        Qupot = compute_Qupot(hourly_wind_speeds, dt)
        Qspot = 0.5 * T * Swe  # Snowfall-limited transport [kg/m]
        Srwe = theta * Swe    # Relocated water equivalent [mm]
        
        if Qupot > Qspot:
            Qinf = 0.5 * T * Srwe
            control = "Snowfall controlled"
        else:
            Qinf = Qupot
            control = "Wind controlled"
        
        Qt = Qinf * (1 - 0.14 ** (F / T))
        
        return {
            "Qupot (kg/m)": Qupot,
            "Qspot (kg/m)": Qspot,
            "Srwe (mm)": Srwe,
            "Qinf (kg/m)": Qinf,
            "Qt (kg/m)": Qt,
            "Control": control
        }   

    def plot_rose(avg_sector_values, overall_avg):
        """
        Plot a 16-sector wind-rose using Plotly instead of Matplotlib.

        Parameters:
        avg_sector_values: list of 16 transport values (kg/m)
        overall_avg: mean Qt (kg/m)
        """

        # Convert to tonnes / m
        avg_tonnes = np.array(avg_sector_values) / 1000.0

        # Sector directions (16 bins)
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        
        # Angles centered on each sector bin
        theta = np.linspace(0, 360, 16, endpoint=False)

        # Build Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Barpolar(
            r=avg_tonnes,
            theta=theta,
            width=[360/16]*16,
            marker_line_color="black",
            marker_line_width=1,
            opacity=0.8,
        ))

        overall_tonnes = overall_avg/1000.0

        fig.update_layout(
            title=(
                f"Average Directional Distribution of Snow Transport<br>"
                f"Overall Average Qt: {overall_tonnes:,.1f} tonnes/m"
            ),
            polar=dict(
                angularaxis=dict(
                    tickmode="array",
                    tickvals=theta,
                    ticktext=directions,
                    rotation=90, # North at the top
                    direction="clockwise"
                )
            ),
            showlegend=False,
            margin=dict(l=30, r=30, t=80, b=30)
        )

        return fig
    # --------------------------------------------------------

    #open Meteo Api data 

    @st.cache_data
    def load_snow_drift_data(lat, lon, start_date, end_date, variables=["temperature_2m", "precipitation", "wind_speed_10m", "wind_direction_10m"]):
        url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly="
        for var in variables:
            url += f"{var}," if var != variables[-1] else f"{var}"
        url += "&timezone=Europe%2FOslo"

        print(f"Fetching data from URL: {url}")
        response = rquests.get(url)
        if response.status_code == 200:
            data = response.json()
        
            hourly_data = data.get("hourly", {})
            df_api = pd.DataFrame(hourly_data)

            # time to column 
            df_api["time"] = pd.to_datetime(df_api["time"], errors="coerce")

            return df_api
        else:
            st.error(f"Feil ved henting av data fra API: {response.status_code}")
            return None
        
    st.title("Snow drift/season - Tabler (2003)")

    if "selected_location" not in st.session_state:
        st.warning("Ingen Koordinater valgt fra kartet. Gå til 'Map'-siden og klikk på et sted for å velge posisjon.")
        st.stop()

    lat, lon = st.session_state["selected_location"]
    st.write(f"Valgt posisjon: Latitude {lat:.4f}, Longitude {lon:.4f}")


    #velger år range
    year = list(range(2021, 2025))
    start_year, end_year = st.select_slider(
        "Velg år for analyse:",
        options=year,
        value=(2021, 2024)
    )

    st.write(f"Valgt år: {start_year} to {end_year}")

    #parametere hentet fra snow_drift.py
    T = 3000
    F =30000
    theta = 0.5

    resultat = []
    secction_verdier = []

    for year in range(start_year, end_year +1):
        start_date = f"{year}-07-01"
        end_date = f"{year + 1}-06-30"
        #add spinner "waiting time" 
        with st.spinner(f"Henter og behandler data for sesongen {year}-{year + 1}..."):
             df_api = load_snow_drift_data(lat, lon, start_date, end_date)
             time.sleep(1)  # Simulerer ventetid for bedre brukeropplevelse
             st.snow()

        
    



        if df_api is None or df_api.empty:
            st.warning(f"Ingen data funnet for sesongen {year}-{year + 1}. Hopper over dette året.")
            continue

        df_api["Swe_hourly"]= df_api.apply(
            lambda row: row["precipitation"] if row["temperature_2m"] <= 1 else 0,
            axis=1
        )

        total_Swe = df_api["Swe_hourly"].sum()
        hourly_wind_speeds = df_api["wind_speed_10m"].tolist()
        hourly_wind_dirs = df_api["wind_direction_10m"].tolist()

        resultata = compute_snow_transport(T, F, theta, total_Swe, hourly_wind_speeds)
        Qt = resultata["Qt (kg/m)"]

        sector_values = compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs)
        resultat.append({"Season": f"{year}-{year+1}", "Qt (kg/m)": Qt})
        secction_verdier.append(sector_values)

    #vis resultater
    if not resultat:
        st.error("Ingen gyldige data funnet for de valgte årene og posisjonen.")
        st.stop()


    result_df = pd.DataFrame(resultat)

    Col1, Col2 = st.columns(2)


    with Col1:
        st.subheader("Snow drift/season Results")
        fig =go.Figure()
        fig.add_trace(go.Scatter
            (x=result_df["Season"], y=result_df["Qt (kg/m)"],
            mode="lines+markers",
            name="Qt"
            ))
        
        fig.update_layout(
            title="Mean Snow Transport per Season",
            xaxis_title="Season",
            yaxis_title="Qt (kg/m)",
            
        )
        st.plotly_chart(fig, use_container_width=True)


    with Col2:
        #wind rose plot
        st.subheader("Wind Rose")
        avg_sector_values = np.mean(secction_verdier, axis=0)
        Qt_avg = result_df["Qt (kg/m)"].mean()

        rose_fig = plot_rose(avg_sector_values, Qt_avg)
        st.plotly_chart(rose_fig, use_container_width=True)


elif page == "Sliding Window Correlation 📉":
    st.title("Sliding Window Correlation Analyse 📉")

    @st.cache_data(ttl=6000)
    def load_mongo_data():
        uri = st.secrets["MongoDB"]["uri"]
        database = st.secrets["MongoDB"]["database"]
  
        
        client = MongoClient(uri, server_api=ServerApi("1"))
        db = client[database]
        
        # Load production data
        prod_data = list(db["production_data"].find())
        prod_data_df = pd.DataFrame(prod_data)
        # Load consumption data
        cons_data = list(db["consumption_data"].find())
        cons_data_df = pd.DataFrame(cons_data)

        # Convert timestamp
        if 'starttime' in prod_data_df.columns:
            prod_data_df['starttime'] = pd.to_datetime(prod_data_df['starttime'])

        return prod_data_df, cons_data_df
    
    @st.cache_data(ttl=6000)
    def load_data_from_api(lat, lon,year, variables =None):
        if variables is None:
            variables = ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]
        url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={year}-01-01&end_date={year}-12-31&hourly="
        for var in variables:
            url += f"{var}," if var != variables[-1] else f"{var}"
        url += "&timezone=Europe%2FOslo"

        print(f"Fetching data from URL: {url}")
        response = rquests.get(url)
        if response.status_code != 200:
           st.error(f"Feil ved henting av data fra API: {response.status_code}")
           return None
        
        data = response.json()
        hourly_data = pd.DataFrame(data["hourly"])
        hourly_data["time"] = pd.to_datetime(hourly_data["time"])

        return hourly_data
    
     #sjekker hvis mongo data er lastet
    if "mongo_data" not in st.session_state:
        prod_data_df, cons_data_df = load_mongo_data()
        st.session_state["mongo_data"] = prod_data_df, cons_data_df
        st.write("Data lastet fra MongoDB.")

    else: 
        prod_data_df, cons_data_df = st.session_state["mongo_data"]
        st.write("Bruker cachet data fra MongoDB.")




    #bruker valg 
    if "selected_location" not in st.session_state:
        st.write("Ingen prisområde valgt. Gå til map-siden og trykke på et område.")
        st.stop()


    st.subheader("Innstillinger for korrelasjonsanalyse")

    Lat , lon = st.session_state["selected_location"]


    col1, col2 = st.columns(2)


    with col1:
        selected_year = st.radio("Year", [2021, 2022, 2023,2024], index=0,horizontal=True)
        selected_price_area = st.radio("Select Price Area", ["NO1", "NO2", "NO3", "NO4", "NO5"], index=0, horizontal=True)
        energy_section = st.radio("Energy Section", ["production", "consumption"], index=0, horizontal=True
                                  )
        
        meteorological_variable = st.selectbox("Select Meteorological Variable",
                                                  ["temperature_2m", "precipitation", "wind_speed_10m",
                                                   "wind_gusts_10m", "wind_direction_10m"])
        Month_n = {
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"   
        }
        
        Month_select = st.selectbox("Select Month",
                                    options=list(Month_n.keys()),
                                    format_func=lambda x: Month_n[x]
                                    )
    with col2:
        window_size = st.slider("Select Sliding Window Size (hours)", 24, 720, 168)
        step_size = st.slider("Select Step Size (hours)", -168, 168, 0)
        Centered = st.slider("Center index for highlighting --> ", 0, 8760, 4380)


    #legg data i dataframe og merge
    meteorological_data= load_data_from_api(Lat, lon, selected_year, variables=[meteorological_variable])    
    Energi_data = prod_data_df if energy_section == "production" else cons_data_df

    #rense energi data
    Energi_data = Energi_data[ (Energi_data["pricearea"] == selected_price_area) & (Energi_data["starttime"].dt.year == selected_year)]    
    Energi_data = Energi_data.groupby("starttime")["quantitykwh"].sum().reset_index()
    Energi_data.rename(columns={"starttime": "time"}, inplace=True)

    meteorological_data.rename(
    columns={meteorological_variable: "meteorological_variable"},
    inplace=True
        )
    merged_data = pd.merge(meteorological_data[["time", "meteorological_variable"]],
                            Energi_data[["time", "quantitykwh"]],
                            on="time",
                            how="inner")
    #filter month
    merged_data = merged_data[merged_data["time"].dt.month == Month_select]
    merged_data = merged_data.sort_values("time").reset_index(drop=True)

    
    merged_data = merged_data.sort_values("time").reset_index(drop=True)

    #legg til flere lag
    merged_data["Meteorological_lag"] = merged_data["meteorological_variable"].shift(step_size)


    def sliding_window_correlation(hourly_data, step_size=0, window_size=168, Centered=4000):

        

        energi = hourly_data["quantitykwh"].reset_index(drop=True)
        meteo = hourly_data["meteorological_variable"].reset_index(drop=True)

        # Lag lagget meteorologisk serie
        meteorological_lag = meteo.shift(step_size)

        # Sliding window correlation
        window_corr = energi.rolling(window_size, center=True).corr(meteorological_lag)
        #progess bar    
        progreess_bar = st.progress(0)
        N= len(energi)
        for i in range(N):
            progreess_bar.progress((i + 1) / N)

        # PLOTTING
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.1,
            subplot_titles=(
                "Meteorologisk data (lagget)",
                "Energi data",
                f"Korrelasjon (vindu: {window_size} timer)"
            )
        )

        # ---- PLOT 1: METEOROLOGI ----
        fig.add_trace(go.Scatter(
            x=list(range(len(meteorological_lag))),
            y=meteorological_lag,
            mode="lines",
            name=f"{meteorological_variable} (lag={step_size})"
        ), row=1, col=1)

        # Markér aktiver vindu
        start = max(0, Centered - window_size // 2 + step_size)
        end = min(len(meteorological_lag), Centered + window_size // 2 + step_size)

        fig.add_trace(go.Scatter(
            x=list(range(start, end)),
            y=meteorological_lag[start:end],
            mode="lines",
            line=dict(color="red", width=4),
            name="Vindu"
        ), row=1, col=1)

        # ---- PLOT 2: ENERGI ----
        fig.add_trace(go.Scatter(
            x=energi.index,
            y=energi,
            mode="lines",
            name="Energi (kWh)"
        ), row=2, col=1)

        start_win = max(0, Centered - window_size // 2)
        end_win = min(len(energi), Centered + window_size // 2)

        fig.add_trace(go.Scatter(
            x=list(range(start_win, end_win)),
            y=energi[start_win:end_win],
            mode="lines",
            line=dict(color="red", width=4)
        ), row=2, col=1)

        # ---- PLOT 3: KORRELASJON ----
        fig.add_trace(go.Scatter(
            x=window_corr.index,
            y=window_corr,
            mode="lines",
            name="Korrelasjon"
        ), row=3, col=1)

        # Markér senterpunkt
        center_idx = Centered + step_size
        if 0 <= center_idx < len(window_corr):
            fig.add_trace(go.Scatter(
                x=[center_idx],
                y=[window_corr[center_idx]],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Senter"
            ), row=3, col=1)

        # Total korrelasjon for hele perioden
        mask = ~np.isnan(meteorological_lag)
        if mask.sum() > 2:
            overall_corr = np.corrcoef(energi[mask], meteorological_lag[mask])[0, 1]
        else:
            overall_corr = np.nan

        return fig, overall_corr

    
    fig, overall_corr = sliding_window_correlation(
        merged_data,
        step_size=step_size,
        window_size=window_size,
        Centered=Centered)
    

    st.plotly_chart(fig)
    st.write(f"Correlation at lag {step_size} hours: {overall_corr:.4f}")



elif page == "Forecasting 📅":
    st.header("Forecasting of energy production and consumption")
    @st.cache_data(ttl=6000)
    def load_mongo_data():
        uri = st.secrets["MongoDB"]["uri"]
        database = st.secrets["MongoDB"]["database"]
        
        client = MongoClient(uri, server_api=ServerApi("1"))
        db = client[database]
        
        # Load production data
        prod_data = list(db["production_data"].find())
        prod_data_df = pd.DataFrame(prod_data)
        # Load consumption data
        cons_data = list(db["consumption_data"].find())
        cons_data_df = pd.DataFrame(cons_data)

        # Convert timestamp
        if 'starttime' in prod_data_df.columns:
            prod_data_df['starttime'] = pd.to_datetime(prod_data_df['starttime'])
        if 'starttime' in cons_data_df.columns:
            cons_data_df['starttime'] = pd.to_datetime(cons_data_df['starttime'])

        return prod_data_df, cons_data_df
    

    #sjekker hvis mongo data er lastet
    if "mongo_data" not in st.session_state:
        prod_data_df, cons_data_df = load_mongo_data()
        st.session_state["mongo_data"] = prod_data_df, cons_data_df
        st.write("Data lastet fra MongoDB.")

    else:
        prod_data_df, cons_data_df = st.session_state["mongo_data"]
        st.write("Bruker cachet data")

    #all elements 
    #   VALG – dataset, område, gruppe, treningsperiode osv.
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Målvariable")
        data_type = st.radio("Select Data Type", ["production", "consumption"], index=0, horizontal=True)
        energi_df = prod_data_df if data_type == "production" else cons_data_df


        price_area = st.radio("Select Price Area", ["NO1", "NO2", "NO3", "NO4", "NO5"], index=0, horizontal=True)

        if data_type == "production":
            energy_group = st.selectbox("Select energy group", ["hydro", "wind", "solar", "thermal", "other"])
        elif data_type == "consumption":
            energy_group = st.selectbox("Select Consumption Type", ["primary", "secondary", "household", "cabin", "tertiary"])


    with col2:
        Date_min = dt.date(2021, 1, 1)
        Date_max = dt.date(2024, 12, 31)
        Date_temp = dt.date(2021, 12, 31)

        st.subheader("trening forecast model")
        start_train= st.date_input("Select start date for training",value= Date_min, min_value=Date_min, max_value=Date_max)
        end_train = st.date_input("Select end date for training", value= Date_temp, min_value=Date_min, max_value=Date_max)

        #legg til error hvis start dato er større enn end dato
        if start_train > end_train:
            st.error("Sluttdato må være etter startdato")
            
        forecast_horizon = st.number_input("Forecast horizon (hours)", 1,1000,168)

    with col3:
        st.subheader("SARIMAX Parametere")
        p = st.number_input("p", 0, 5,1)
        d = st.number_input("d", 0, 2,1)
        q = st.number_input("q", 0, 5,1)


    with col4:
        st.subheader("Sesongkomponent")
        P = st.number_input("Seasonal AR order (P)", 0, 3,1)
        D = st.number_input("Seasonal differencing order (D)", 0, 2,0)
        Q = st.number_input("Seasonal MA order (Q)", 0, 3,1)
        S = st.number_input("Seasonal period (S)", 1, 8760,24)

    #velger exogenous variables
    exogenous_v =[]
    try:
        #gjør om brei kolonne slikt at de kan preseneteres samtidig kategorisk
        if data_type == "production":
            piovt = (energi_df.reset_index()
                     .pivot_table(index='starttime', columns=["productiongroup","pricearea"], values='quantitykwh', aggfunc='mean')
            )
            piovt_cols = [f"{grp}_{area}" for grp, area in piovt.columns]

        elif data_type == "consumption":
            piovt = (energi_df.reset_index()
                     .pivot_table(index='starttime', columns=["consumptiontype","pricearea"], values='quantitykwh', aggfunc='mean')
            )
            piovt_cols = [f"{grp}_{area}" for grp, area in piovt.columns]
                

        else:
            cols =[]


        selected_col = f"{energy_group}_{price_area}"

        groups_with_samee_area = [col for col in piovt_cols if col.endswith(f"_{price_area}") and col != selected_col]
        group_with_another_area = [col for col in piovt_cols if col.startswith(f"{energy_group}_") and not col.endswith(f"_{price_area}")]

        exogenous_list = groups_with_samee_area + group_with_another_area
    except Exception:
        exogenous_list = []
    exogenous_v = st.multiselect(" Exogenous Variables,", options=exogenous_list)

    #kjør forecasting modell
    kjor_forecast = st.button("Kjør Forecasting Model")

    if kjor_forecast:
            # Bred tabell for alle grupper/områder
            if data_type == "production":
                full = (prod_data_df.reset_index()
                           .pivot_table(index='starttime', columns=['productiongroup','pricearea'],values ='quantitykwh', aggfunc='mean')
                )

            else:
                full = (cons_data_df.reset_index()
                           .pivot_table(index='starttime', columns=['consumptiontype','pricearea'],values ='quantitykwh', aggfunc='mean')
                )


            full.columns = [f"{grp}_{area}" for grp, area in full.columns]
            full = full.sort_index().asfreq("H")

#            #fyll manglende verdier
            full = full.ffill().interpolate(limit=24)

        #definer target kolonne
            selected_col = f"{energy_group}_{price_area}"
            if selected_col not in full.columns:
                st.error(f"Valgte kolonne '{selected_col}' finnes ikke i dataene.")
                st.stop()

                # definer treningsperiode
            train_data = pd.to_datetime(start_train)
            train_end = pd.to_datetime(end_train) + pd.Timedelta(hours=23,minutes=59,seconds=59)

            y_train= full[selected_col].loc[train_data:train_end].copy()

            if y_train.empty:
                st.error("Ingen treningsdata funnet for den valgte perioden.")
                st.stop()

             # eksogene datasett
             #behold kun kolonner som faktisk eksisterer
            eksog_selected = [col for col in exogenous_v if col in full.columns]
            if eksog_selected:
                x_train = full[eksog_selected].loc[train_data:train_end].copy()
                x_train = x_train.ffill().interpolate(limit=24).fillna(method="bfill").fillna(0)
            else:
                x_train = None


            # Bygg og tren SARIMAX-modellen
            model = sm.tsa.statespace.SARIMAX(y_train,
                                              order=(p,d,q),
                                              seasonal_order=(P,D,Q,S),
                                              exog=x_train,
                                              enforce_stationarity=False,
                                              enforce_invertibility=False,
                                              )
            
            try:
                with st.spinner("Tilpasser SARIMAX-modell..."):
                    model_fit = model.fit(disp=False)
            except Exception as e:
                st.error(f"Feil under modelltilpasning: {e}")
                st.stop()

            with st.expander("Vis modell sammendrag"):
                st.text(model_fit.summary())

            # Prognose
            H = int(forecast_horizon)           
            siste_tid = y_train.index[-1]
            fremtidige_tider = pd.date_range(start=siste_tid + pd.Timedelta(hours=1), periods=H, freq='H')


            if eksog_selected:
             #sikre at eksog er en liste 
             if isinstance(eksog_selected, str):
                 eksog_selected = [eksog_selected]
            #hent siste tilgjengelige eksogene data
             siste_eksog = full.loc[:train_end, eksog_selected].iloc[-1]
             if not isinstance(siste_eksog, pd.Series):
                siste_eksog = pd.Series([siste_eksog], index=eksog_selected)

             future_exog = pd.DataFrame([siste_eksog.values] * H, columns=eksog_selected, index=fremtidige_tider)


                
            else:
                future_exog = None

            prognose = model_fit.get_forecast(steps=H, exog=future_exog)
            prognose_mean = prognose.predicted_mean 
            prognose_ci = prognose.conf_int()

            #plotting 
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=y_train.index,
                y=y_train,
                mode="lines",
                name="Treningsdata"
            ))

            fig.add_trace(go.Scatter(
                x=prognose_ci.index,
                y=prognose_ci.iloc[:, 0],
                mode="lines",
                name ="forecast"
            ))

            fig.add_trace(go.Scatter(
                x=prognose_ci.index,
                y=prognose_ci.iloc[:, 1],
                mode="lines",
                fill='tonexty',
                line=dict(width=0),
                name="Konfidensintervall"
            ))


            st.plotly_chart(fig, use_container_width=True)