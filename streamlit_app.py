import streamlit as st
import pandas as pd
import re
import json

# Page configuration
st.set_page_config(
    page_title="Text Classifier",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .main-header {
        font-family: 'Georgia', serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #e94560;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-family: 'Georgia', serif;
        color: #a2a8d3;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(233, 69, 96, 0.1);
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .dict-section {
        background: rgba(162, 168, 211, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stTextArea textarea {
        background: rgba(26, 26, 46, 0.8) !important;
        border: 1px solid rgba(162, 168, 211, 0.3) !important;
        color: #e8e8e8 !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(26, 26, 46, 0.8) !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Text Dictionary Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your dataset and classify text using customizable dictionaries</p>', unsafe_allow_html=True)

# Initialize session state for dictionaries
if 'dictionaries' not in st.session_state:
    st.session_state.dictionaries = {
        'urgency_marketing': {
            'limited', 'limited time', 'limited run', 'limited edition', 'order now',
            'last chance', 'hurry', 'while supplies last', "before they're gone",
            'selling out', 'selling fast', 'act now', "don't wait", 'today only',
            'expires soon', 'final hours', 'almost gone'
        },
        'exclusive_marketing': {
            'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
            'members only', 'vip', 'special access', 'invitation only',
            'premium', 'privileged', 'limited access', 'select customers',
            'insider', 'private sale', 'early access'
        }
    }

# Classification function
def classify_text(text, dictionaries):
    text_lower = str(text).lower()
    results = {}
    for dict_name, terms in dictionaries.items():
        matches = [term for term in terms if re.search(r'\b' + re.escape(term) + r'\b', text_lower)]
        results[dict_name] = 1 if matches else 0
        results[f'{dict_name}_matches'] = ', '.join(matches) if matches else ''
    return results

# Create two columns for layout
col1, col2 = st.columns([1, 1])

# Left column - Dictionary Management
with col1:
    st.markdown("### 📚 Dictionary Management")
    
    # Add new dictionary
    with st.expander("➕ Add New Dictionary", expanded=False):
        new_dict_name = st.text_input("Dictionary Name (no spaces, use underscores)", key="new_dict_name")
        new_dict_terms = st.text_area(
            "Terms (one per line)", 
            height=150,
            key="new_dict_terms",
            help="Enter each term on a new line"
        )
        if st.button("Add Dictionary", type="primary"):
            if new_dict_name and new_dict_terms:
                clean_name = new_dict_name.strip().replace(' ', '_').lower()
                terms_set = {term.strip().lower() for term in new_dict_terms.split('\n') if term.strip()}
                st.session_state.dictionaries[clean_name] = terms_set
                st.success(f"Added dictionary: {clean_name} with {len(terms_set)} terms")
                st.rerun()
            else:
                st.warning("Please provide both a name and terms")
    
    # Edit existing dictionaries
    st.markdown("#### Current Dictionaries")
    
    for dict_name in list(st.session_state.dictionaries.keys()):
        with st.expander(f"📖 {dict_name} ({len(st.session_state.dictionaries[dict_name])} terms)"):
            # Display current terms
            current_terms = '\n'.join(sorted(st.session_state.dictionaries[dict_name]))
            edited_terms = st.text_area(
                "Edit terms (one per line)",
                value=current_terms,
                height=200,
                key=f"edit_{dict_name}"
            )
            
            col_save, col_delete = st.columns(2)
            with col_save:
                if st.button("💾 Save Changes", key=f"save_{dict_name}"):
                    terms_set = {term.strip().lower() for term in edited_terms.split('\n') if term.strip()}
                    st.session_state.dictionaries[dict_name] = terms_set
                    st.success(f"Updated {dict_name}")
                    st.rerun()
            
            with col_delete:
                if st.button("🗑️ Delete Dictionary", key=f"delete_{dict_name}", type="secondary"):
                    del st.session_state.dictionaries[dict_name]
                    st.success(f"Deleted {dict_name}")
                    st.rerun()

# Right column - File Upload and Processing
with col2:
    st.markdown("### 📤 Upload & Classify")
    
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file with text data to classify"
    )
    
    if uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        
        st.markdown("#### Preview of uploaded data")
        st.dataframe(df.head(), use_container_width=True)
        
        # Column selection
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if text_columns:
            selected_column = st.selectbox(
                "Select the text column to classify",
                options=text_columns,
                help="Choose which column contains the text you want to analyze"
            )
            
            # Dictionary selection
            selected_dicts = st.multiselect(
                "Select dictionaries to use",
                options=list(st.session_state.dictionaries.keys()),
                default=list(st.session_state.dictionaries.keys()),
                help="Choose which dictionaries to apply"
            )
            
            if st.button("🚀 Run Classification", type="primary", use_container_width=True):
                if selected_dicts:
                    # Filter dictionaries
                    active_dicts = {k: v for k, v in st.session_state.dictionaries.items() if k in selected_dicts}
                    
                    # Apply classification
                    with st.spinner("Classifying text..."):
                        classification_results = df[selected_column].apply(
                            lambda x: classify_text(x, active_dicts)
                        )
                        results_df = pd.DataFrame(classification_results.tolist())
                        
                        # Combine with original data
                        df_output = pd.concat([df, results_df], axis=1)
                    
                    st.success("Classification complete!")
                    
                    # Summary Statistics
                    st.markdown("#### 📊 Classification Summary")
                    
                    summary_cols = st.columns(len(selected_dicts))
                    for idx, dict_name in enumerate(selected_dicts):
                        with summary_cols[idx]:
                            count = df_output[dict_name].sum()
                            percentage = count / len(df) * 100
                            st.metric(
                                label=dict_name.replace('_', ' ').title(),
                                value=f"{count} matches",
                                delta=f"{percentage:.1f}%"
                            )
                    
                    # Results preview
                    st.markdown("#### 📋 Results Preview")
                    st.dataframe(df_output, use_container_width=True)
                    
                    # Download button
                    csv_output = df_output.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Classified Results (CSV)",
                        data=csv_output,
                        file_name="classified_output.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("Please select at least one dictionary")
        else:
            st.warning("No text columns found in the uploaded file")

# Sidebar with instructions
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. **Upload** your CSV file with text data
    2. **Select** the column containing text to analyze
    3. **Customize** dictionaries in the left panel:
       - Edit existing terms
       - Add new dictionaries
       - Delete unwanted dictionaries
    4. **Run** the classification
    5. **Download** your results
    """)
    
    st.markdown("---")
    st.markdown("## 💡 Tips")
    st.markdown("""
    - Terms are case-insensitive
    - Use word boundaries for matching
    - Add multi-word phrases as needed
    - Export results include match details
    """)
    
    st.markdown("---")
    
    # Export/Import dictionaries
    st.markdown("## 🔄 Dictionary Backup")
    
    # Export
    dict_export = {k: list(v) for k, v in st.session_state.dictionaries.items()}
    st.download_button(
        "📤 Export Dictionaries (JSON)",
        data=json.dumps(dict_export, indent=2),
        file_name="dictionaries.json",
        mime="application/json",
        use_container_width=True
    )
    
    # Import
    uploaded_dict = st.file_uploader("📥 Import Dictionaries (JSON)", type=['json'])
    if uploaded_dict is not None:
        try:
            imported = json.load(uploaded_dict)
            st.session_state.dictionaries = {k: set(v) for k, v in imported.items()}
            st.success("Dictionaries imported successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error importing: {str(e)}")
