import streamlit as st
import pandas as pd

# Konfigurer side
st.set_page_config(page_title="IND320 App", layout="wide")

# --- Data lasting med caching ---
@st.cache_data
def load_data():
    return pd.read_csv("open-meteo-subset.csv")

# Last data 
try: 
    df = load_data()
except FileNotFoundError:
    df = None

# --- Sidebar navigasjon ---
st.sidebar.title("Navigasjon")
page = st.sidebar.radio("Velg side:", ["Hjem", "Data-tabell", "Plot", "Side 4"])

# --- Side 1: Hjem ---
if page == "Hjem":
    st.title("IND320 – Compulsory Work 1")
    st.write("""
    Velkommen til min Streamlit-app!  
    Denne appen viser data fra **open-meteo-subset.csv** og gir mulighet til å utforske tabeller og grafer.
    """)

elif page == "Data-tabell":
    st.header("Data-tabell")
    if df is not None:
        # Lag en kolonne "month" for å kunne filtrere
        df["time"] = pd.to_datetime(df["time"])
        df["month"] = df["time"].dt.to_period("M").astype(str)

        # Ta ut første måned
        first_month = sorted(df["month"].unique())[0]
        mdf = df[df["month"] == first_month]

        # Bygg en tabell med én rad per variabel
        rows = []
        for col in df.columns:
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
    else:
        st.error("Fant ikke datafilen 'open-meteo-subset.csv'.")

elif page == "Plot":
    st.header("Visualisering av data")

    if df is not None:
        # Sørg for at time er datetime og legg til month
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["month"] = df["time"].dt.to_period("M").astype(str)

        # Velg kolonne
        numeric_cols = [c for c in df.columns if c not in ("time", "month")]
        col_choice = st.selectbox("Velg kolonne:", ["Alle"] + numeric_cols)

        # Velg måneder (slider)
        months = sorted(df["month"].unique())
        month_range = st.select_slider(
            "Velg måned(er):",
            options=months,
            value=(months[0], months[0])  # default = første måned
        )

        # Filtrer datasett
        if isinstance(month_range, tuple):
            subset = df[(df["month"] >= month_range[0]) & (df["month"] <= month_range[1])]
        else:
            subset = df[df["month"] == month_range]

        st.subheader(f"Plot for {col_choice} ({month_range})")

        # Tegn plott
        if col_choice == "Alle":
            st.line_chart(
                subset.set_index("time")[numeric_cols],
                use_container_width=True
            )
        else:
            st.line_chart(
                subset.set_index("time")[[col_choice]],
                use_container_width=True
            )

    else:
        st.error("Fant ikke datafilen 'open-meteo-subset.csv'.")

# --- Side 4: Dummy ---
elif page == "Side 4":
    st.header("Side 4 (plassholder)")
    st.write("Her kan du legge til mer innhold senere.")
