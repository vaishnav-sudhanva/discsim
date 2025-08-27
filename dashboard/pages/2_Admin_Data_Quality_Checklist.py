import streamlit as st
import pandas as pd
from src.utils.state_management import initialize_states, reset_session_states, reset_upload
from src.utils.admin_data_quality_checklist.helpers.file_upload import handle_file_upload
from src.utils.admin_data_quality_checklist.helpers.preliminary_tests import run_preliminary_tests
from src.utils.admin_data_quality_checklist.helpers.functionality_map import execute_functionality, sidebar_functionality_select
from src.utils.utility_functions import set_page_config,setFooter,setheader

set_page_config()

def admin_data_quality_check():
    # File selection
    file_option = st.sidebar.radio("Choose an option:", ("Upload a new file", "Select a previously uploaded file"))

    # Initialize states
    initialize_states()

    # Clear relevant session state when switching options
    if st.session_state.previous_file_option != file_option:
        st.session_state.uploaded_file = None
        st.session_state.uploaded_file_id = None
        reset_session_states()
        st.session_state.previous_file_option = file_option

    uploaded_file = handle_file_upload(file_option, category="admin_data_quality_checklist")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_file = uploaded_file
        if uploaded_file != st.session_state.previous_uploaded_file:
            reset_session_states()
            st.session_state.previous_uploaded_file = uploaded_file
        st.sidebar.divider()
        # Run preliminary tests
        if run_preliminary_tests(uploaded_file):
            
            # Sidebar for functionality selection
            functionality = sidebar_functionality_select()
            
            # Use the selected functionality
            st.session_state.navbar_selection = functionality
                        
            execute_functionality(functionality, uploaded_file, df)

    else:
        st.info("Please upload a CSV file to begin.")
        reset_session_states()
        st.session_state.previous_uploaded_file = None

if st.session_state.get('reset_upload', False):
    reset_upload()
    st.rerun()

if __name__ == "__main__":
    selectedNav = setheader("Admin Data Quality")
    if selectedNav == "Pre Survey":
          st.switch_page("pages/1_Pre_Survey.py")
    if selectedNav == "Post Survey":
          st.switch_page("pages/3_Post_Survey.py")
    admin_data_quality_check()

    setFooter()