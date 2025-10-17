import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="IND320 App", layout="wide")

@st.cache_data
def load_data():
     # Laster data fra CSV-fil
    return pd.read_csv("open-meteo-subset.csv")

# Prøv å laste data
try:
    df = load_data()
except FileNotFoundError:
    df = None

# Navigasjon
st.sidebar.title("Navigasjon")
page = st.sidebar.radio("Velg side:", ["Hjem", "Data-tabell", "Plot", "Side 4"])

# hjemmeside
if page == "Hjem":
    st.title("IND320 – Compulsory Work 1")
    st.write(
        "Denne appen viser data fra **open-meteo-subset.csv** og gir enkel utforsking av tabeller og grafer."
    )

# DATA-TABELL
elif page == "Data-tabell":
    st.header("Data-tabell")
    if df is None:
        st.error("Fant ikke datafilen 'open-meteo-subset.csv'.")
    else:
        d = df.copy()
        d["time"] = pd.to_datetime(d["time"])
        d["month"] = d["time"].dt.to_period("M").astype(str)

        first_month = sorted(d["month"].unique())[0]
        mdf = d[d["month"] == first_month]

        rows = []
        for col in d.columns:
            if col not in ("time", "month"):
                rows.append({
                    "Variable": col,
                    "Preview (first month)": mdf[col].dropna().tolist()
                })
        table = pd.DataFrame(rows)

        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "Variable": st.column_config.TextColumn("Variable"),
                "Preview (first month)": st.column_config.LineChartColumn("Preview"),
            },
            use_container_width=True,
        )
        st.caption(f"Viser første måned: {first_month}")

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
           

# --- DUMMY ---
elif page == "Side 4":
    st.header("Side 4 (Dummy)")
    st.write("Plassholder for fremtidig innhold.")
