import streamlit as st

def render_sidebar_filters(df):
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50) # Optional logo
    st.sidebar.title("Validata BI Controls")
    
    # 1. Biological Indicator
    indicators = df['Indicator'].unique().tolist()
    indicator = st.sidebar.selectbox("1. Select Indicator:", indicators)
    df_filtered = df[df['Indicator'] == indicator]
    
    # 2. Target Percentile (Fraud bracket)
    if 'Target_Percentile' in df_filtered.columns:
        pcts = sorted(df_filtered['Target_Percentile'].unique().tolist())
        # Default to 30% if it exists, otherwise pick the first one
        default_pct = "30%" if "30%" in pcts else pcts[0]
        selected_pct = st.sidebar.selectbox("2. Target Evaluation Bracket:", pcts, index=pcts.index(default_pct))
        df_filtered = df_filtered[df_filtered['Target_Percentile'] == selected_pct]

    st.sidebar.divider()
    
    # 3. L1 Budget Filter
    st.sidebar.header("📍 Supervisor (L1)")
    l1_budgets = sorted(df_filtered['L1_Budget_Pct'].unique().tolist())
    selected_l1_budgets = st.sidebar.multiselect("Filter L1 Budgets:", l1_budgets, default=l1_budgets)
    if selected_l1_budgets:
        df_filtered = df_filtered[df_filtered['L1_Budget_Pct'].isin(selected_l1_budgets)]
        
    # 4. L2 Budget Filter
    st.sidebar.header("🔎 Auditor (L2)")
    l2_budgets = sorted(df_filtered['L2_Budget_Pct'].unique().tolist())
    selected_l2_budgets = st.sidebar.multiselect("Filter L2 Budgets:", l2_budgets, default=l2_budgets)
    if selected_l2_budgets:
        df_filtered = df_filtered[df_filtered['L2_Budget_Pct'].isin(selected_l2_budgets)]
        
    return df_filtered, indicator, selected_pct