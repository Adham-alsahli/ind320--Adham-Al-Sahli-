import streamlit as st
import pandas as pd

# Konfigurer side
st.set_page_config(page_title="IND320 App", layout="wide")

# --- Data lasting med caching ---
@st.cache_data
def load_data():
    # Husk å legge "open-meteo-subset.csv" i prosjektmappa
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

# --- Side 2: Data-tabell ---
elif page == "Data-tabell":
    st.header("Data-tabell")
    if df is not None:
        st.dataframe(
            df,
            column_config={
                col: st.column_config.LineChartColumn()
                for col in df.columns[1:]
            },
            hide_index=True,
        )
    else:
        st.error("Fant ikke datafilen 'open-meteo-subset.csv'. Last opp eller legg den i prosjektmappa.")

# --- Side 3: Plot ---
elif page == "Plot":
    st.header("Visualisering av data")

    if df is not None:
        # Velg kolonne
        col_choice = st.selectbox("Velg kolonne:", ["Alle"] + list(df.columns[1:]))

        # Velg måneder
        months = df[df.columns[0]].unique()
        month_choice = st.select_slider("Velg måned:", options=months, value=months[0])

        # Filtrer på valgt måned
        subset = df[df[df.columns[0]] == month_choice]

        st.subheader(f"Plot for {col_choice} – måned: {month_choice}")

        if col_choice == "Alle":
            st.line_chart(subset.set_index(df.columns[0]))
        else:
            st.line_chart(subset.set_index(df.columns[0])[[col_choice]])
    else:
        st.error("Fant ikke datafilen 'open-meteo-subset.csv'.")

# --- Side 4: Dummy ---
elif page == "Side 4":
    st.header("Side 4 (plassholder)")
    st.write("Her kan du legge til mer innhold senere.")
