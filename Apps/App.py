import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from pymongo.server_api import ServerApi
import matplotlib.pyplot as plt
st.set_page_config(page_title="IND320 App", layout="wide")

@st.cache_data
def load_data():
     # Laster data fra CSV-fil
    return pd.read_csv("Ind320\Data\open-meteo-subset.csv")

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
    st.title("IND320 - Compulsory Work 1")
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
           

elif page == "Side 4":
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











