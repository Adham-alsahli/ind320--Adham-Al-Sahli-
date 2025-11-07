import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from pymongo.server_api import ServerApi
import matplotlib.pyplot as plt
from pathlib import Path
import tomllib
from scipy.fftpack import dct, idct
from sklearn.neighbors import LocalOutlierFactor


# Robust import av STL
HAS_STATSMODELS = False
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
except ModuleNotFoundError:
    # vi håndterer dette senere i stl_decomposition()
    pass

from scipy.signal import spectrogram
import requests as rquests


# Ensure this script is run via `streamlit run` so Streamlit runtime (session/context) is available.
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

# Navigasjon
st.sidebar.title("Navigasjon")
page = st.sidebar.radio("Velg side:", ["Hjem","MongoDB","New_A", "Data-tabell", "Plot", "New_B"])

# hjemmeside
if page == "Hjem":
    st.title("IND320 - Compulsory Work 3")
    st.write(
        "Denne appen viser data fra API-en og gir enkel utforsking av tabeller og grafer."
    )

elif page == "MongoDB":
    st.header("Side 4 - Elhub produksjon 2021")

    #  Mongo tilkobling
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi

    @st.cache_resource # Cache for å unngå reconnect hver gang
    def load_data_from_mongo():
        uri = st.secrets["MongoDB"]["uri"]
        database = st.secrets["MongoDB"]["database"]
        collection = st.secrets["MongoDB"]["collection"]
            
        client = MongoClient(uri, server_api=ServerApi("1"))
        db = client[database]
        col = db[collection]
        # Hent alle dokumenter fra samlingen
        data = list(col.find({}, {"_id": 0}))  # dropp _id i query
        df_local = pd.DataFrame(data)

        # Sikre riktige feltnavn og typer
        
        time_col = "starttime" if "starttime" in df_local.columns else ("start_time" if "start_time" in df_local.columns else None)
        area_col = "pricearea" if "pricearea" in df_local.columns else ("price_area" if "price_area" in df_local.columns else None)
        group_col = "productiongroup" if "productiongroup" in df_local.columns else ("production_group" if "production_group" in df_local.columns else None)
        qty_col = "quantitykwh" if "quantitykwh" in df_local.columns else ("quantity_kwh" if "quantity_kwh" in df_local.columns else None)

        if not all([time_col, area_col, group_col, qty_col]):
            raise RuntimeError("Fant ikke forventede feltnavn i Mongo-dokumentene (starttime/pricearea/productiongroup/quantitykwh).")

        # Konverter tid til datetime (UTC) før .dt brukes
        df_local[time_col] = pd.to_datetime(df_local[time_col], utc=True, errors="coerce")
        # Dropp rader uten tid/område/gruppe
        df_local = df_local.dropna(subset=[time_col, area_col, group_col])

        return df_local, time_col, area_col, group_col, qty_col

    df, time_col, area_col, group_col, qty_col = load_data_from_mongo()

    #Layout: to kolonner 
    left_column, right_column = st.columns(2)

    # VENSTRE: Radio + Pie 
    with left_column:
        st.subheader("Production Share by Group")
        price_areas = sorted(df[area_col].unique())
        selected_area = st.radio("Select Price Area", price_areas, horizontal=True)

        area_data = df[df[area_col] == selected_area]
        s = area_data.groupby(group_col)[qty_col].sum().sort_values(ascending=False)

        # lag to små kolonner: venstre = pai, høyre = tabell/legend
        pcol, tcol = st.columns([2, 1])

        with pcol:
            fig, ax = plt.subplots(figsize=(6, 6))
            # ingen labels og ingen autopct på selve paien
            wedges, _ = ax.pie(s.values, labels=None, autopct=None)
            ax.set_title(f"Production Distribution by Group in {selected_area} (2021)")
            st.pyplot(fig)

        with tcol:
            pct = (s / s.sum() * 100).round(1)
            # hent farger fra pai og bygg tabell
            colors = [w.get_facecolor() for w in wedges]
            tbl = pd.DataFrame({
                "group": s.index,

                "%": pct.values,
                "color": colors
            })

           # Hjelpefunksjon for å fargelegge rader
            def highlight_color(row):
                color = colors[row.name]
                rgba = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})"
                # bakgrunnsfarge + hvit tekst
                return [f"background-color: {rgba}; color: white" for _ in row.index]

            styled = tbl.drop(columns=["color"]).style.apply(highlight_color, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)


           

    # HØYRE: Pills/Multiselect + Måned + Line
    with right_column:
        st.subheader("Monthly Production Trend")

        # Velg produksjonsgrupper
        all_groups = sorted(df[group_col].dropna().unique().tolist())
        if hasattr(st, "pills"):
            selected_group = st.pills("Select Production Group", all_groups, selection_mode="multi", default=all_groups)
        else:
            selected_group = st.multiselect("Select Production Group", all_groups, default=all_groups[:3])

        # Velg måned
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        selected_month = st.selectbox("Select Month", months)
        month_idx = months.index(selected_month) + 1

        # Filtrer
        filtered_data = df[
            (df[time_col].dt.month == month_idx) &
            (df[area_col] == selected_area) &
            (df[group_col].isin(selected_group))
        ].copy()
    # Line plot
        if not filtered_data.empty:
            pivot_data = filtered_data.pivot_table(
                values=qty_col,
                index=time_col,
                columns=group_col,
                aggfunc="sum"
            ).sort_index().fillna(0.0)

            fig, ax = plt.subplots(figsize=(9, 5))
            for colname in pivot_data.columns:
                ax.plot(pivot_data.index, pivot_data[colname], label=colname)
            ax.set_xlabel("Time")
            ax.set_ylabel("Production (kWh)")
            ax.set_title(f"Hourly Production by Group in {selected_area} - {selected_month} 2021")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Ingen data for valgt kombinasjon.")

    # Expander 
    with st.expander("Data Source"):
        st.markdown("""
Dataene som vises på denne siden kommer fra **Elhub sitt Energy Data API** (https://api.elhub.no/),  
som gir timesdata for produksjon i ulike energigrupper på tvers av norske prisområder.  

For mer informasjon, se [Elhub API Services](https://api.elhub.no/).

        """)

# New_A-side: STL og Spectrogram
elif page == "New_A":
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
    def load_mongo_data():# Load data from MongoDB
        with open("C:\\Users\\adham\\Documents\\Nmbu\\5\\ind320\\IND320-main\\Ind320\\.streamlit\\secrets.toml", "rb") as f:
            cfg = tomllib.load(f)
#         Laster inn brukernavn og passord for MongoDB fra secrets.toml
        USR = cfg["MongoDB"]["username"]
        PWD = cfg["MongoDB"]["pwd"]



        #USR, PWD = open("../.streamlit/secrects.toml")["MongoDB"].read().splitlines()
        uri = f"mongodb+srv://{USR}:{PWD}@adham.j1syfjw.mongodb.net/?retryWrites=true&w=majority&appName=IND320-Adham"

        # Creating a new client and connecting to server
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client["IND320_assignment_2"]
        collection = db["production_per_group_hour"]

        data = list(collection.find())
        elhub_dataBase = pd.DataFrame(data)
        # Convert time column to datetime if needed
        if "starttime" in elhub_dataBase.columns:
            elhub_dataBase["starttime"] = pd.to_datetime(elhub_dataBase["starttime"])

        return elhub_dataBase
#stL and spectrogram code functions

    def stl_decomposition(df, price_area = "NO5", production_group = "solar", period = 24, seasonal = 7, trend = None, robust = True):
        # If statsmodels is not installed, show a friendly message figure instead of raising an import error
        if not HAS_STATSMODELS:
            # Show a small figure with an instruction and also display a Streamlit message
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "Missing dependency: statsmodels\nSee instruction below to install.", ha="center", va="center")
            ax.axis('off')
            try:
                import sys
                install_cmd = f"{sys.executable} -m pip install statsmodels"
                # show command in Streamlit UI for easy copy/paste
                st.error("Missing dependency: statsmodels. Install into the Python used by Streamlit:")
                st.code(install_cmd)
            except Exception:
                # If Streamlit not available or sys can't be read, silently continue
                pass
            return fig

        data = df[(df['pricearea'] == price_area) & (df['productiongroup'] == production_group)]
        ts = data["quantitykwh"]
        ts.index = pd.to_datetime(data["starttime"])
        ts.sort_index(inplace=True) # to ensure time series is sorted

        # Validate and coerce STL parameters: trend must be an odd integer >=3 and > period
        try:
            period = int(period)
        except Exception:
            period = 24

        if trend is None:
            # choose smallest odd integer > period
            candidate = period + 1
            if candidate % 2 == 0:
                candidate += 1
            trend = max(3, candidate)
        else:
            try:
                trend = int(trend)
            except Exception:
                trend = None

        if trend is None or trend <= period or trend < 3 or (trend % 2 == 0):
            # compute a valid fallback and warn the user in the UI
            candidate = period + 1
            if candidate % 2 == 0:
                candidate += 1
            st.warning(f"Invalid STL trend={trend} for period={period}; using trend={candidate} instead.")
            trend = max(3, candidate)

        stl = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust)
        result = stl.fit()

        #plot
        fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        ax[0].plot(ts, label='Original')
        ax[0].set_title(f"{production_group} Production ({price_area}) - Original")
        ax[0].legend()

        ax[1].plot(result.trend, label='Trend', color='orange')
        ax[1].set_title('Trend Component')
        ax[1].legend()

        ax[2].plot(result.seasonal, label='Seasonal', color='green')
        ax[2].set_title('Seasonal Component')
        ax[2].legend()


        ax[3].plot(result.resid, label='Residual', color='red')
        ax[3].set_title('Residual Component')
        ax[3].legend()
        plt.tight_layout()

        return fig
    def create_spectrogram(df, price_area="NO2", production_group="hydro",
                       window_length=500, overlap=130, cmap='plasma'):

        data = df[(df['pricearea'] == price_area) & (df['productiongroup'] == production_group)]
        ts = data["quantitykwh"]
        ts.index = pd.to_datetime(data["starttime"])
        ts.sort_index(inplace=True)

        # Compute spectrogram
        f, t, Sxx = spectrogram(ts, fs=1, nperseg=window_length, noverlap=overlap)

        # Plot spectrogram
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading="nearest", cmap=cmap)
        ax.set_ylabel('Frequency [cycles per hour]')
        ax.set_xlabel('Time [hours]')
        ax.set_title(f'Spectrogram of {production_group} Production ({price_area})')
        fig.colorbar(im, ax=ax, label="Power [dB]")
        plt.tight_layout()

        return fig
    elhub_dataBase = load_mongo_data()
    with tabel1:
        st.header("STL Decomposition")
        selected_group_stl = st.radio("Select Production Group for Analysis", st.session_state["selected_group"])
        fig = stl_decomposition(elhub_dataBase, price_area=selected_area, production_group=selected_group_stl)
        st.pyplot(fig)

    with tabel2:
        st.header("Spectrogram Analysis")
        selected_group_spec = st.radio("Select Production Group for Spectrogram", st.session_state["selected_group"], key="spec")
        fig = create_spectrogram(elhub_dataBase, price_area=selected_area, production_group=selected_group_spec)
        st.pyplot(fig)


# DATA-TABELL
elif page == "Data-tabell":
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
elif page == "Plot":
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
elif page == "New_B":
    st.header("New_B Page")
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
#apply high-pass filter
        satv  = high_pass_filter(temperature, freq_cutoff)

        # compute robust statistics
        median = np.median(satv)
        mad = np.median(np.abs(satv - median))
        threshold = 1.4826 * mad

        # defining SPC and control limits
        upper_control_limit = median + num_std * threshold
        lower_control_limit = median - num_std * threshold
        upper_curve = temperature + (upper_control_limit - satv)
        lower_curve = temperature + (lower_control_limit - satv)

        # detect outliers
        outliers = (satv > upper_control_limit) | (satv < lower_control_limit)
        outliers_indices = np.where(outliers)[0]
        #plot temp and outliers
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(time, temperature, label='Temperature', color='blue', alpha=0.7)
        ax.scatter(time[outliers], temperature[outliers], color='red', label='Outliers')
        plt.plot(time, upper_curve, color='g', linestyle='--', label='Upper Control Limit')
        plt.plot(time, lower_curve, color='green', linestyle='--', label='Lower Control Limit')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Outliers Detected via Robust SPC')
        ax.legend()
        plt.tight_layout()
        
        #preparing summary of outliers
        Summary_outliers = {
            "outlier_indices": outliers_indices,
            "outlier_times": time[outliers],
            "outlier_temperatures": temperature[outliers],
            "num_outliers": len(outliers_indices),
            "upper_control_limit": upper_control_limit,
            "lower_control_limit": lower_control_limit,
            "Robust_median": median,
            "robust_std": threshold
        }
        return fig , Summary_outliers     

    def detect_lof_outliers(time, data_variable,n_neighbors=20, contamination=0.01):
    # Reshape data for LOF
        D = np.array(data_variable).reshape(-1, 1)
    
        # Initialize LOF model
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        lof_predict = lof.fit_predict(D)   
        scores = -lof.negative_outlier_factor_ # Get LOF scores and higher negative scores indicate more abnormal points

            
        
        
        # Identify outlier indices
        outlier = lof_predict == -1
        outlier_indices = np.where(outlier)[0]
        #plot precipitation and outliers
        fig, ax  = plt.subplots(figsize=(15, 6))
        ax.plot(time, data_variable, label='selected_variable', color='blue', alpha= 0.7)
        ax.scatter(time[outlier], data_variable[outlier], color='red', label='Outliers')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Anomalies Detected via Local Outlier Factor (LOF)')
        ax.legend()
        plt.tight_layout()
        #preparing summary of outliers
        summary_outliers = {
            "outlier_indices": outlier_indices,
            "outlier_times": time[outlier],
            "outlier_selected_variable": data_variable[outlier],
            "num_outliers": len(outlier_indices),
            "lof_scores": scores[outlier]
        }
        return fig, summary_outliers
        
    if "weather_data" in st.session_state:
        weather_data = st.session_state["weather_data"]
        st.write("Data loaded from API.")

    else:
        weather_data = load_data_from_api(lat, lon, selected_year,variables=["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"])
        st.session_state["weather_data"] = weather_data
        st.write("Data loaded from API.")


    with tabel1: # Outlier/SPC analysis on temperature data
        st.subheader("Outlier/SPC Analysis on Temperature Data")
        selected_area = "temperature_2m"
        freq_cutoff = st.slider("Select Frequency Cutoff for High-Pass Filter:", min_value=10, max_value=500, value=100, step=1)
        num_std = st.slider("Select Number of Standard Deviations for Control Limits:", min_value=1, max_value=5, value=3, step=1)

        fig, summary_outliers = detect_temperature_outliers(
            weather_data["time"],
            weather_data[selected_area].values,
            freq_cutoff=freq_cutoff,
            num_std=num_std
        )
        st.pyplot(fig)
        st.subheader(f"Summary of Detected Outliers ({summary_outliers['num_outliers']} total)")
        st.write(f"Robust Median: {summary_outliers['Robust_median']:.2f}")
    
    with tabel2:#
        st.subheader("Anomaly/LOF Analysis on Selected Data")
        # Select variable for LOF analysis
        selected_area = st.radio("Select Variable for LOF Analysis", ["precipitation", "wind_speed_10m", "wind_gusts_10m"])
        contamination = st.slider("Select Contamination Level for LOF:", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        n_neighbors = st.slider("Select Number of Neighbors for LOF:", min_value=5, max_value=50, value=20, step=1)
        # Detect outliers using LOF
        fig, summary_outliers = detect_lof_outliers(
            weather_data["time"].values,
            weather_data[selected_area].values,
            n_neighbors=n_neighbors,
            contamination=contamination
        )
        st.pyplot(fig)
       