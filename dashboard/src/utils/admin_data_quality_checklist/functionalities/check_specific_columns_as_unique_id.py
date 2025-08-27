import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

UNIQUE_ID_CHECK_ENDPOINT = f"{API_BASE_URL}/unique_id_check"

def check_specific_columns_as_unique_id(df):
    st.session_state.drop_export_rows_complete = False
    st.session_state.drop_export_entries_complete = False
    title_info_markdown = """
        Use this feature to check whether the column(s) you think form the unique ID is indeed the unique ID.
        - Verifies if selected column(s) can serve as a unique identifier for the dataset.
        - You can select up to 4 columns to check.
        - The function will return whether the selected column(s) can work as a unique ID.
        - Valid input format: CSV file
        - A minimum of ONE column must be selected.
    """
    st.markdown("<h2 style='text-align: center;'>Check Specific Columns as Unique ID</h2>", unsafe_allow_html=True, help=title_info_markdown)
    
    columns = st.multiselect("Select columns to check", options=df.columns.tolist())
    
    if columns and st.button("Check Unique ID"):
        with st.spinner("Checking unique ID..."):
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
            data = df_clean.where(pd.notnull(df_clean), None).to_dict('records')
            payload = {"data": data, "columns": columns}
            response = requests.post(UNIQUE_ID_CHECK_ENDPOINT, json=payload)
            
            if response.status_code == 200:
                result = response.json()['result']
                if result[1]:
                    st.success("Check completed!")
                    st.write(result[0])
                else:
                    st.warning(result[0])
                    st.write("Go to Unique ID Verifier to check for column uniqueness")
            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"Error: {response.status_code} - {error_detail}")