import os
import streamlit as st
import pandas as pd
import requests
import traceback
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

FIND_UNIQUE_IDS_ENDPOINT = f"{API_BASE_URL}/find_unique_ids"

def unique_id_verifier(uploaded_file):
    st.session_state.drop_export_rows_complete = False
    st.session_state.drop_export_entries_complete = False
    title_info_markdown = """
        Use this feature to let the system identify the list of unique IDs in the dataset.
        - Numerical columns, or combinations which are comprised of more numerical columns, will be given precedence while displaying the output.
        - If you have also used any of the other modules before, you can also use the same dataset used there by clicking the "Use existing dataset" button below.
        - Valid input format for dataset: xlsx or csv
        - A minimum of ONE column has to be selected
        - Max no. of selectable columns: As many as the number of column headers
    """
    st.markdown("<h2 style='text-align: center;'>Unique ID Verifier</h2>", unsafe_allow_html=True, help=title_info_markdown)

    if st.button("Find Unique IDs"):
        with st.spinner("Finding unique IDs..."):
            try:
                uploaded_file.seek(0)
                files = {"file": ("uploaded_file.csv", uploaded_file, "text/csv")}
                response = requests.post(FIND_UNIQUE_IDS_ENDPOINT, files=files)
                
                if response.status_code == 200:
                    unique_ids = response.json()
                    
                    if unique_ids:
                        st.success("Unique IDs found!")
                        st.write("The best possible Column(s) or Combinations that can act as Unique ID are as follows:")
                        df = pd.DataFrame(unique_ids)
                        df['UniqueID'] = df['UniqueID'].apply(lambda x: ' + '.join(x))
                        df = df.rename(columns={'UniqueID': 'Unique ID (data type)', 'Numeric_DataTypes': 'Is Numeric'})
                        st.dataframe(df['Unique ID (data type)'], use_container_width=True, hide_index=True)
                    else:
                        st.warning("No unique identifiers found. All columns or combinations have at least one duplicate.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    st.write("Response content:", response.content)
                    st.write("Response headers:", response.headers)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Traceback:", traceback.format_exc())