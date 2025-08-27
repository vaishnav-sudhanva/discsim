import streamlit as st
import pandas as pd
from src.utils.state_management import (
    initialize_states,
    reset_session_states,
    reset_upload,
)
from src.utils.post_survey_analysis.helpers.file_upload import handle_file_upload
from src.utils.post_survey_analysis.functionality import execute_post_survey_analysis
from src.utils.utility_functions import set_page_config,setFooter,setheader

set_page_config()

def post_survey_analysis():

    st.sidebar.header("ECD Nested Supervision")
    
    # File selection
    file_option = st.sidebar.radio(
        "Choose an option:",
        ("Upload a new file", "Select a previously uploaded file"),
    )

    # Initialize states
    initialize_states()

    # Define page-specific session state keys
    previous_file_option_key = "previous_file_option_post_survey_analysis"
    uploaded_file_key = "uploaded_file_post_survey_analysis"
    previous_uploaded_file_key = "previous_uploaded_file_post_survey_analysis"
    reset_upload_key = "reset_upload_post_survey_analysis"

    # Clear relevant session state when switching options
    if st.session_state.get(previous_file_option_key) != file_option:
        st.session_state[uploaded_file_key] = None
        st.session_state.uploaded_file_id = None
        reset_session_states()
        st.session_state[previous_file_option_key] = file_option

    uploaded_file = handle_file_upload(
        file_option, category="post_survey_analysis"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state[uploaded_file_key] = uploaded_file
        if uploaded_file != st.session_state.get(previous_uploaded_file_key):
            reset_session_states()
            st.session_state[previous_uploaded_file_key] = uploaded_file

        execute_post_survey_analysis(uploaded_file, df)

    else:
        st.info("Please upload a CSV file to begin.")
        reset_session_states()
        st.session_state[previous_uploaded_file_key] = None

    if st.session_state.get(reset_upload_key, False):
        reset_upload()
        st.rerun()

if __name__ == "__main__":
    selectedNav = setheader("Post Survey")
    if selectedNav == "Pre Survey":
          st.switch_page("pages/1_Pre_Survey.py")
    if selectedNav == "Admin Data Quality":
          st.switch_page("pages/2_Admin_Data_Quality_Checklist.py")
    post_survey_analysis()

    setFooter()