import json
import os
import traceback
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from dotenv import load_dotenv
from src.utils.admin_data_quality_checklist.helpers.graph_functions import plot_pie_chart

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

MISSING_ENTRIES_ENDPOINT = f"{API_BASE_URL}/missing_entries"

def missing_entries_analysis(uploaded_file, df):
    st.session_state.drop_export_rows_complete = False
    st.session_state.drop_export_entries_complete = False    
    title_info_markdown = """
        This function returns the count and percentage of missing values for a given variable, with optional filtering and grouping by a categorical variable.
        - Analyzes the dataset to find missing entries in a specified column.
        - Optionally groups or filters the analysis by other categorical columns.
        - Provides a table of rows with missing entries.
        - Valid input format: CSV file
    """
    st.markdown("<h2 style='text-align: center;'>Missing Entries Analysis</h2>", unsafe_allow_html=True, help=title_info_markdown)
    col1, col2, col3 = st.columns(3)
    with col1:
        column_to_analyze = st.selectbox("Select column to analyze for missing entries:", options=df.columns.tolist(), index=0)
    with col2:
        group_by = st.selectbox("Group by (optional)", options=["None"] + df.columns.tolist(), index=0, help="Analyze missing entries within distinct categories of another column. This is useful if you want to understand how missing values are distributed across different groups.")
    with col3:
        filter_by_col = st.selectbox("Filter by column (optional)", options=["None"] + df.columns.tolist(), index=0, help="Focus on a specific subset of your data by selecting a specific value in another column. This is helpful when you want to analyze missing entries for a specific condition.")
    
    col4, col5, col6 = st.columns(3)
    if filter_by_col != "None":
        with col4:
            filter_by_value = st.selectbox("Filter value", df[filter_by_col].unique().tolist())
        with col5:
            st.write("")
        with col6:
            st.write("")
    else:
        filter_by_value = None

    # Analyze Missing Entries
    if st.button("Analyze Missing Entries"):
        with st.spinner("Analyzing missing entries..."):
            try:
                uploaded_file.seek(0)  # Reset file pointer
                files = {"file": ("uploaded_file.csv", uploaded_file, "text/csv")}
                payload = {
                    "column_to_analyze": column_to_analyze,
                    "group_by": group_by if group_by != "None" else None,
                    "filter_by": {filter_by_col: filter_by_value} if filter_by_col != "None" else None
                }
                response = requests.post(
                    MISSING_ENTRIES_ENDPOINT,
                    files=files,
                    data={"input_data": json.dumps(payload)}
                )
                                                    
                if response.status_code == 200:
                    result = response.json()
                    
                    if result["grouped"]:
                        st.write("Missing entries by group:")
                        grouped_data = []
                        for group, (count, percentage) in result["analysis"].items():
                            grouped_data.append({
                                group_by: group,  # Use the name of the group-by column
                                "Missing Count": count,
                                "Missing Percentage": f"{percentage:.2f}%" if percentage is not None else "N/A"
                            })
                        
                        grouped_df = pd.DataFrame(grouped_data)
                        grouped_df = grouped_df.sort_values("Missing Count", ascending=False)
                        
                        # Center-align just Missing Count and Missing Percentage
                        st.dataframe(grouped_df.style.set_properties(**{
                            'text-align': 'center',
                            'text': 'center',
                            'align-items': 'center',
                            'justify-content': 'center'
                        }, subset=['Missing Count', 'Missing Percentage']), use_container_width=True, hide_index=True)
                        
                        # Create a 100% stacked column chart
                        data = pd.DataFrame([(group, percentage, 100-percentage if percentage is not None else 0) 
                                            for group, (count, percentage) in result["analysis"].items()],
                                            columns=[group_by, 'Missing', 'Present'])
                        data = data.sort_values('Missing', ascending=False)
                        fig = px.bar(data, x=group_by, y=['Missing', 'Present'], 
                                    title=f"Missing vs Present Entries by {group_by}",
                                    labels={'value': 'Percentage', 'variable': 'Status'},
                                    color_discrete_map={'Missing': 'red', 'Present': 'green'},
                                    text='value')
                        fig.update_layout(barmode='relative', yaxis_title='Percentage')
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                        st.plotly_chart(fig)
                    else:
                        count, percentage = result["analysis"]
                        if percentage is not None:
                            st.write(f"Missing entries: {count} ({percentage:.2f}%)")
                            labels = ['Missing', 'Present']
                            values = [percentage, 100-percentage]
                            fig = plot_pie_chart(labels, values, "Missing vs Present Entries (%)")
                            st.plotly_chart(fig)
                        else:
                            st.write(f"Missing entries: {count} (percentage unavailable)")
                    
                    if result["filtered"]:
                        st.info(f"Results are filtered by {filter_by_col} = {filter_by_value}")
                        
                    # Display the table of missing entries
                    if "missing_entries_table" in result:
                        if not result["missing_entries_table"]:
                            st.warning("The missing entries table is empty.")
                        else:
                            missing_entries_df = pd.DataFrame(result["missing_entries_table"])
                                                                    
                            if column_to_analyze in missing_entries_df.columns:
                                missing_entries_df = missing_entries_df.sort_values(column_to_analyze, ascending=False)
                                st.success(f"Sorted by column: '{column_to_analyze}'")
                            else:
                                st.warning(f"Column '{column_to_analyze}' not found in the missing entries table. Displaying unsorted data.")
                            
                            with st.expander("Rows with Missing Entries:"):
                                st.dataframe(missing_entries_df, use_container_width=True, hide_index=True)
                    else:
                        st.error("The 'missing_entries_table' key is not present in the API response.")

                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {response.status_code} - {error_detail}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Traceback:", traceback.format_exc())
