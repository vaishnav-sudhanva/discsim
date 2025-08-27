import streamlit as st
from src.utils.admin_data_quality_checklist.functionalities.unique_id_verifier import unique_id_verifier
from src.utils.admin_data_quality_checklist.functionalities.check_specific_columns_as_unique_id import check_specific_columns_as_unique_id
from src.utils.admin_data_quality_checklist.functionalities.drop_export_duplicate_entries import drop_export_duplicate_entries
from src.utils.admin_data_quality_checklist.functionalities.drop_export_duplicate_rows import drop_export_duplicate_rows
from src.utils.admin_data_quality_checklist.functionalities.missing_entries_analysis import missing_entries_analysis
from src.utils.admin_data_quality_checklist.functionalities.zero_entries_analysis import zero_entries_analysis
from src.utils.admin_data_quality_checklist.functionalities.indicator_fill_rate_analysis import indicator_fill_rate_analysis
from src.utils.admin_data_quality_checklist.functionalities.frequency_table_analysis import frequency_table_analysis

FUNCTIONALITY_MAP = {
    "Unique ID Verifier": {
        "function": unique_id_verifier,
        "keywords": [],
        "requires_df": False
    },
    "Check Specific Columns as Unique ID": {
        "function": check_specific_columns_as_unique_id,
        "keywords": [],
        "requires_df": "only_df"
    },
    "Drop/Export Duplicate Entries": {
        "function": drop_export_duplicate_entries,
        "keywords": ["duplicate"],
        "requires_df": True
    },
    "Drop/Export Duplicate Rows": {
        "function": drop_export_duplicate_rows,
        "keywords": ["duplicate"],
        "requires_df": False
    },
    "Missing Entries Analysis": {
        "function": missing_entries_analysis,
        "keywords": ["missing"],
        "requires_df": True
    },
    "Zero Entries Analysis": {
        "function": zero_entries_analysis,
        "keywords": ["zero"],
        "requires_df": True
    },
    "Indicator Fill Rate Analysis": {
        "function": indicator_fill_rate_analysis,
        "keywords": [],
        "requires_df": True
    },
    "Frequency Table Analysis": {
        "function": frequency_table_analysis,
        "keywords": [],
        "requires_df": True
    }
}

def get_relevant_functionality(warning):
    warning = warning.lower()
    for functionality, info in FUNCTIONALITY_MAP.items():
        if any(keyword in warning for keyword in info["keywords"]):
            return functionality
    return "Unique ID Verifier"  # Default functionality

def sidebar_functionality_select():
    st.sidebar.header("Select Functionality")
    functionality = st.sidebar.pills(
        label="Choose a functionality",
        options=list(FUNCTIONALITY_MAP.keys()),
        default=list(FUNCTIONALITY_MAP.keys())[0],
        key="functionSelectAdmin",
        label_visibility="collapsed"
    )
    st.session_state.navbar_selection = functionality
    return functionality

def execute_functionality(functionality, uploaded_file, df=None):
    if st.session_state.navbar_selection == functionality:
        func_info = FUNCTIONALITY_MAP[functionality]
        if func_info["function"] == check_specific_columns_as_unique_id:
            return func_info["function"](df)
        elif func_info["function"] != check_specific_columns_as_unique_id and func_info["requires_df"]:
            return func_info["function"](uploaded_file, df)
        else:
            return func_info["function"](uploaded_file)