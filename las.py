"""
RiML - Rice Machine Learning Application
ROBUST VERSION with comprehensive error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback

# Page configuration
st.set_page_config(
    page_title="RiML - Rice Stress Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 5.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-card {
        background-color: #E8F5E9;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #2E7D32;
        margin: 15px 0;
    }
    .warning-card {
        background-color: #FFF9C4;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FBC02D;
        margin: 15px 0;
    }
    .error-card {
        background-color: #FFEBEE;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #E53935;
        margin: 15px 0;
    }
    .success-card {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 15px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 30px;
        border-radius: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2E7D32 0%, #558B2F 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ROBUST MODEL LOADING with error handling
@st.cache_resource
def load_model_components():
    """Load model with comprehensive error handling"""
    errors = []
    
    try:
        with open('Pretrained_rice_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        errors.append("❌ Pretrained_rice_model.pkl not found")
        model = None
    except Exception as e:
        errors.append(f"❌ Error loading model: {str(e)}")
        model = None
    
    try:
        with open('SCALER.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        errors.append("❌ SCALER.pkl not found")
        scaler = None
    except Exception as e:
        errors.append(f"❌ Error loading scaler: {str(e)}")
        scaler = None
    
    try:
        with open('Top_genes.pkl', 'rb') as f:
            top_genes = pickle.load(f)
    except FileNotFoundError:
        errors.append("❌ Top_genes.pkl not found")
        top_genes = None
    except Exception as e:
        errors.append(f"❌ Error loading genes: {str(e)}")
        top_genes = None
    
    try:
        with open('MODEL_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
    except:
        model_info = {
            'cv_accuracy': 0.917,
            'training_accuracy': 1.0,
            'n_features': 200,
            'classes': ['Control', 'Drought', 'Salt', 'Cold']
        }
    
    return model, scaler, top_genes, model_info, errors

# ROBUST DATA PREPROCESSING
def preprocess_uploaded_data(df, top_genes):
    """
    Preprocess uploaded data with extensive error checking
    Returns: (df_selected, sample_names, available_genes, warnings, errors)
    """
    warnings = []
    errors = []
    
    try:
        # Step 1: Identify gene ID column
        gene_col_candidates = ['ID_REF', 'Gene', 'gene', 'GENE', 'Gene_ID', 'gene_id', 'ID']
        gene_col = None
        
        for col in gene_col_candidates:
            if col in df.columns:
                gene_col = col
                break
        
        if gene_col is None:
            # Assume first column is gene IDs
            gene_col = df.columns[0]
            warnings.append(f"⚠️ No standard gene ID column found. Using '{gene_col}' as gene IDs.")
        
        # Set index
        df = df.set_index(gene_col)
        
        # Step 2: Check data type
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            errors.append("❌ No numeric columns found. Please check your file format.")
            return None, None, None, warnings, errors
        
        # Use only numeric columns
        df = df[numeric_cols]
        
        # Step 3: Remove AFFX controls
        affx_count = sum(df.index.str.startswith('AFFX-', na=False))
        if affx_count > 0:
            df = df[~df.index.str.startswith('AFFX-', na=False)]
            warnings.append(f"ℹ️ Removed {affx_count} AFFX control probes")
        
        # Step 4: Handle missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            df = df.fillna(df.mean())
            warnings.append(f"ℹ️ Filled {missing_count} missing values with column means")
        
        # Step 5: Check for inf values
        inf_count = np.isinf(df.values).sum()
        if inf_count > 0:
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(df.mean())
            warnings.append(f"ℹ️ Replaced {inf_count} infinite values")
        
        # Step 6: Transpose
        df_transposed = df.T
        
        # Step 7: Check gene overlap
        available_genes = [g for g in top_genes if g in df_transposed.columns]
        missing_genes = set(top_genes) - set(df_transposed.columns)
        
        if len(available_genes) == 0:
            errors.append("❌ None of the model's genes found in your data. Please check gene ID format.")
            return None, None, None, warnings, errors
        
        if len(missing_genes) > 0:
            warnings.append(f"⚠️ {len(missing_genes)} model genes missing from your data ({len(available_genes)}/200 present)")
        
        if len(available_genes) < 50:
            errors.append(f"❌ Too few genes found ({len(available_genes)}/200). Predictions will be unreliable.")
            return None, None, None, warnings, errors
        
        # Step 8: Select genes
        df_selected = df_transposed[available_genes]
        sample_names = df_selected.index.tolist()
        
        # Step 9: Check sample count
        if len(sample_names) == 0:
            errors.append("❌ No samples found in data")
            return None, None, None, warnings, errors
        
        if len(sample_names) > 1000:
            warnings.append(f"⚠️ Large number of samples ({len(sample_names)}). Processing may take time.")
        
        return df_selected, sample_names, available_genes, warnings, errors
        
    except Exception as e:
        errors.append(f"❌ Preprocessing error: {str(e)}")
        return None, None, None, warnings, errors

# Main header
st.markdown('<p class="main-header">🌾 RiML</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Rice Machine Learning </p>', unsafe_allow_html=True)

# Load model
model, scaler, top_genes, model_info, load_errors = load_model_components()

# Show loading errors if any
if load_errors:
    st.error("### ⚠️ Model Loading Issues:")
    for error in load_errors:
        st.error(error)
    st.info("**Solution:** Make sure these files are in the same directory as the app:")
    st.code("""
Pretrained_rice_model.pkl
SCALER.pkl
Top_genes.pkl
MODEL_info.pkl
    """)
    st.stop()

# Success message
st.success("✅ Model loaded successfully!")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔬 Analysis", "ℹ️ About"])

# =====================================================================
# TAB 1: HOME
# =====================================================================
with tab1:
    st.markdown("## 🎯 Welcome to RiML")
    
    # Added "About RiML" passage here
    st.markdown("""
    <div class="info-card">
        <p><b>RiML</b> is a web-based machine learning application for classifying rice stress conditions from gene expression data. 
        Built with <b>Streamlit</b> and powered by a <b>Random Forest classifier</b>, RiML helps researchers identify 
        <b>drought</b>, <b>salt</b>, <b>cold</b>, and <b>control</b> conditions in rice samples.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h2>91.7%</h2>
                <p>CV Accuracy</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h2>200</h2>
                <p>Feature Genes</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h2>4</h2>
                <p>Stress Classes</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>🤖 Machine Learning Model</h3>
        <p><b>Algorithm:</b> Random Forest Classifier</p>
        <p><b>Architecture:</b> 100 Decision Trees (max_depth=5)</p>
        <p><b>Validation:</b> Leave-One-Out Cross-Validation</p>
        <p><b>Dataset:</b> GSE6901 (Gene Expression Omnibus)</p>
        <p><b>Performance:</b> 91.7% CV Accuracy, 100% Training</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
        <h3>🌾 Stress Types Detected</h3>
        <p><b>🌱 Control:</b> Normal, unstressed conditions</p>
        <p><b>🏜️ Drought:</b> Water deficit stress</p>
        <p><b>🧂 Salt:</b> High salinity stress</p>
        <p><b>❄️ Cold:</b> Low temperature stress</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>📋 How to Use</h3>
        <ol>
            <li>Go to <b>Analysis</b> tab</li>
            <li>Upload your CSV/TSV file</li>
            <li>Click <b>ANALYZE</b></li>
            <li>View results & graphs</li>
            <li>Download predictions</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# TAB 2: ANALYSIS
# =====================================================================
with tab2:
    st.markdown("## 🔬 Rice Stress Analysis")
    
    # File format instructions
    with st.expander("📖 File Format Guide - CLICK HERE FOR HELP", expanded=False):
        st.markdown("""
        ### ✅ Required File: Gene Expression Data
        
        **Accepted Formats:**
        - CSV (.csv)
        - TSV/Tab-delimited (.txt, .tsv)
        
        **Required Structure:**
        ```
        ID_REF        Sample1    Sample2    Sample3
        Os01g0100100  245.5      312.8      198.3
        Os01g0100200  567.2      523.9      601.5
        Os02g0100300  123.7      145.2      118.9
        ```
        
        **Requirements:**
        1. **First column:** Gene IDs (can be named: ID_REF, Gene, gene_id, etc.)
        2. **Other columns:** Sample expression values (numeric)
        3. **Rows:** Genes
        4. **Columns:** Samples
        
        ---
        
        ### ✅ Optional File: GPL Annotation
        
        **What is it?**
        - Platform annotation file from GEO
        - Contains gene names, symbols, and descriptions
        - Enhances your results with biological information!
        
        **Example:** `GPL2025-2968.txt`
        
        **Format:**
        ```
        ID            Gene_Symbol  Description
        Os01g0100100  OsACT1      Actin protein
        Os01g0100200  OsWRKY1     WRKY transcription factor
        ```
        
        **Benefits of uploading GPL:**
        - ✅ See gene names (not just IDs)
        - ✅ See gene descriptions
        - ✅ Better interpretation of results
        - ✅ Export gene annotations
        
        ---
        
        ### ⚠️ Common Issues:
        - Make sure gene IDs match the format (Os01g...)
        - Ensure expression values are numeric
        - Check that file is properly formatted (no extra commas/tabs)
        - GPL file is optional but recommended for better results!
        """)
    
    st.markdown("---")
    
    # File upload
    col1, col2 = st.columns([2, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload gene expression file (Required)",
            type=['csv', 'txt', 'tsv'],
            help="Upload CSV or TSV file with genes in rows and samples in columns",
            key="expression_file"
        )
    
    with col2:
        gpl_file = st.file_uploader(
            "📋 Upload GPL annotation file (Optional)",
            type=['csv', 'txt', 'tsv'],
            help="Upload GPL file to get gene names and descriptions in results",
            key="gpl_file"
        )
    
    st.markdown("""
    **💡 Tip:** 
    - Expression file: Your gene expression matrix (Required)
    - GPL file: Platform annotation file for gene names/descriptions (Optional - enhances results!)
    """)
    
    if uploaded_file is not None:
        try:
            # Read file with robust handling
            st.info("📂 Reading file...")
            
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    # For GEO series matrix files, skip comment lines starting with !
                    # First try with comment parameter
                    df = pd.read_csv(uploaded_file, sep='\t', comment='!')
            except Exception as e:
                # If that fails, try reading without comment parameter
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, comment='#')
                    else:
                        df = pd.read_csv(uploaded_file, sep='\t', comment='#')
                except:
                    # Last resort: read without any comment handling
                    uploaded_file.seek(0)
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                    else:
                        df = pd.read_csv(uploaded_file, sep='\t', on_bad_lines='skip')
            
            st.success(f"✅ File loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")
            
            # Preview
            with st.expander("👁️ Preview Data (first 10 rows)", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Data validation
            st.markdown("### 🔍 Data Validation")
            
            validation_col1, validation_col2 = st.columns(2)
            
            with validation_col1:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                if numeric_cols > 0:
                    st.success(f"✅ Found {numeric_cols} numeric columns")
                else:
                    st.error("❌ No numeric columns found!")
            
            with validation_col2:
                total_genes = df.shape[0]
                st.info(f"ℹ️ Total genes in file: {total_genes}")
            
            st.markdown("---")
            
            # Analyze button
            if st.button("🚀 ANALYZE DATA", type="primary", use_container_width=True):
                
                # Load GPL annotations if provided
                gpl_annotations = None
                if gpl_file is not None:
                    with st.spinner("📋 Loading GPL annotations..."):
                        try:
                            if gpl_file.name.endswith('.csv'):
                                gpl_df = pd.read_csv(gpl_file, comment='#')
                            else:
                                gpl_df = pd.read_csv(gpl_file, sep='\t', comment='#')
                            
                            # Find relevant columns
                            id_col = None
                            for col in ['ID', 'ID_REF', 'Gene ID', 'Probe ID']:
                                if col in gpl_df.columns:
                                    id_col = col
                                    break
                            
                            if id_col is None:
                                id_col = gpl_df.columns[0]
                            
                            # Find gene symbol and description columns
                            symbol_col = None
                            desc_col = None
                            
                            for col in gpl_df.columns:
                                if 'symbol' in col.lower() or 'gene_symbol' in col.lower():
                                    symbol_col = col
                                if 'title' in col.lower() or 'description' in col.lower() or 'definition' in col.lower():
                                    desc_col = col
                            
                            if symbol_col or desc_col:
                                cols_to_keep = [id_col]
                                if symbol_col:
                                    cols_to_keep.append(symbol_col)
                                if desc_col:
                                    cols_to_keep.append(desc_col)
                                
                                gpl_annotations = gpl_df[cols_to_keep].copy()
                                gpl_annotations.columns = ['Gene_ID', 'Gene_Symbol', 'Description'][:len(cols_to_keep)]
                                gpl_annotations = gpl_annotations.set_index('Gene_ID')
                                
                                st.success(f"✅ GPL annotations loaded: {len(gpl_annotations)} genes")
                            else:
                                st.warning("⚠️ Could not find gene symbol/description columns in GPL file")
                        
                        except Exception as e:
                            st.warning(f"⚠️ Could not load GPL file: {str(e)}")
                            gpl_annotations = None
                
                with st.spinner("🔄 Processing your data..."):
                    
                    # Preprocess
                    df_selected, sample_names, available_genes, warnings, errors = preprocess_uploaded_data(df, top_genes)
                    
                    # Show warnings
                    if warnings:
                        st.markdown("### ⚠️ Processing Warnings:")
                        for warning in warnings:
                            st.warning(warning)
                    
                    # Show errors and stop if critical
                    if errors:
                        st.markdown("### ❌ Processing Errors:")
                        for error in errors:
                            st.error(error)
                        
                        st.markdown("""
                        <div class="error-card">
                        <h4>💡 How to Fix:</h4>
                        <ul>
                            <li>Check that your file has gene IDs in the first column</li>
                            <li>Ensure expression values are numeric (not text)</li>
                            <li>Verify gene ID format matches the model (e.g., Os01g...)</li>
                            <li>Make sure file is properly formatted CSV or TSV</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        st.stop()
                    
                    if df_selected is None:
                        st.error("❌ Could not process data. Please check file format.")
                        st.stop()
                    
                    # Scale data
                    try:
                        X_scaled = scaler.transform(df_selected)
                    except Exception as e:
                        st.error(f"❌ Scaling error: {str(e)}")
                        st.info("This usually means the gene IDs don't match. Please check your data format.")
                        st.stop()
                    
                    # Predict
                    try:
                        predictions = model.predict(X_scaled)
                        probabilities = model.predict_proba(X_scaled)
                        class_names = model.classes_
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.stop()
                    
                    # SUCCESS!
                    st.markdown("---")
                    st.markdown("## 🎉 ANALYSIS COMPLETE!")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📊 Samples Analyzed", len(predictions))
                    
                    with col2:
                        avg_conf = np.mean(np.max(probabilities, axis=1)) * 100
                        st.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")
                    
                    with col3:
                        unique_preds = len(np.unique(predictions))
                        st.metric("🌾 Stress Types", unique_preds)
                    
                    with col4:
                        stress_count = len(predictions) - sum(predictions == 'Control')
                        st.metric("⚠️ Stressed Samples", stress_count)
                    
                    st.markdown("---")
                    
                    # Predictions table
                    st.markdown("### 📋 Detailed Predictions")
                    
                    emoji_map = {
                        'Control': '🌱',
                        'Drought': '🏜️',
                        'Salt': '🧂',
                        'Cold': '❄️'
                    }
                    
                    results = []
                    for sample, pred, prob in zip(sample_names, predictions, probabilities):
                        max_prob = np.max(prob)
                        results.append({
                            'Sample': sample,
                            'Stress': f"{emoji_map.get(pred, '')} {pred}",
                            'Confidence': f"{max_prob*100:.1f}%",
                            'Control %': f"{prob[list(class_names).index('Control')]*100:.1f}",
                            'Drought %': f"{prob[list(class_names).index('Drought')]*100:.1f}",
                            'Salt %': f"{prob[list(class_names).index('Salt')]*100:.1f}",
                            'Cold %': f"{prob[list(class_names).index('Cold')]*100:.1f}"
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display with color coding
                    st.dataframe(results_df, use_container_width=True, height=400)
                    
                    # If GPL annotations available, show gene information
                    if gpl_annotations is not None:
                        st.markdown("---")
                        st.markdown("### 🧬 Gene Information (from GPL file)")
                        
                        # Get top important genes from the data
                        available_genes_in_gpl = [g for g in available_genes if g in gpl_annotations.index]
                        
                        if len(available_genes_in_gpl) > 0:
                            st.info(f"ℹ️ Found annotations for {len(available_genes_in_gpl)}/{len(available_genes)} genes used in prediction")
                            
                            # Show top 20 genes with annotations
                            top_genes_to_show = available_genes_in_gpl[:20]
                            gene_info_list = []
                            
                            for gene in top_genes_to_show:
                                if gene in gpl_annotations.index:
                                    gene_data = {'Gene_ID': gene}
                                    if 'Gene_Symbol' in gpl_annotations.columns:
                                        gene_data['Gene_Symbol'] = gpl_annotations.loc[gene, 'Gene_Symbol']
                                    if 'Description' in gpl_annotations.columns:
                                        desc = str(gpl_annotations.loc[gene, 'Description'])
                                        gene_data['Description'] = desc[:100] + '...' if len(desc) > 100 else desc
                                    gene_info_list.append(gene_data)
                            
                            if gene_info_list:
                                gene_info_df = pd.DataFrame(gene_info_list)
                                st.dataframe(gene_info_df, use_container_width=True, height=300)
                                
                                # Add download for gene info
                                gene_csv = gene_info_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Gene Annotations (CSV)",
                                    gene_csv,
                                    "gene_annotations.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                        else:
                            st.warning("⚠️ No matching genes found between your data and GPL file")
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "riml_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    
                    # Visualizations
                    st.markdown("### 📊 Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart
                        pred_counts = pd.Series(predictions).value_counts()
                        fig_pie = px.pie(
                            values=pred_counts.values,
                            names=pred_counts.index,
                            title="<b>Stress Distribution</b>",
                            hole=0.4,
                            color_discrete_map={
                                'Control': '#4CAF50',
                                'Drought': '#FF9800',
                                'Salt': '#2196F3',
                                'Cold': '#00BCD4'
                            }
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Bar chart
                        fig_bar = px.bar(
                            x=pred_counts.index,
                            y=pred_counts.values,
                            title="<b>Sample Counts</b>",
                            color=pred_counts.index,
                            color_discrete_map={
                                'Control': '#4CAF50',
                                'Drought': '#FF9800',
                                'Salt': '#2196F3',
                                'Cold': '#00BCD4'
                            }
                        )
                        fig_bar.update_layout(showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Heatmap
                    prob_df = pd.DataFrame(
                        probabilities * 100,
                        columns=class_names,
                        index=[s[:15] + '...' if len(s) > 15 else s for s in sample_names]
                    )
                    
                    fig_heat = px.imshow(
                        prob_df.T,
                        labels=dict(x="Sample", y="Stress Type", color="Probability (%)"),
                        title="<b>Probability Heatmap</b>",
                        color_continuous_scale="RdYlGn",
                        text_auto='.0f'
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                    st.success("✅ Analysis complete! All visualizations are ready for publication.")
                    
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())
            st.info("Please check your file format and try again.")

# =====================================================================
# TAB 3: ABOUT
# =====================================================================
with tab3:
    st.markdown("## ℹ️ About RiML")
    
    st.markdown("""
    <div class="info-card">
    <h3>🤖 About the ML Model</h3>
    <p><b>Algorithm:</b> Random Forest Classifier</p>
    <p><b>Trees:</b> 100</p>
    <p><b>Features:</b> 200 stress-responsive genes</p>
    <p><b>Training:</b> GSE6901 dataset (12 samples)</p>
    <p><b>Validation:</b> Leave-One-Out CV</p>
    <p><b>Accuracy:</b> 91.7%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>🙏 Acknowledgments</h3>
        <p>  We would like to express my deepest gratitude to 
        <b>Dr. Kushagrah Kashyap</b> for his invaluable guidance, 
        continuous support, and insightful feedback throughout the development of RiML.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>👥 Team</h3>
        <p><b>Developers:</b> Pranjal Dalvi and Sejal Chaudhari</p>
        <p><b>Institution:</b> DES Pune University</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>📞 Contact</h3>
        <ul>
            <li><b>Email:</b> dpranjal889@gmail.com</li>
            <li><b>LinkedIn:</b> 
                <a href="https://www.linkedin.com/in/pranjal-dalvi-2b6538316" target="_blank">
                Pranjal Dalvi</a>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>🚀 Future Work</h3>
        <ul>
            <li>Support for more stress types such as <b>Heat</b> and <b>UV</b>.</li>
            <li>Integration of <b>RNA-Seq</b> datasets for improved analysis.</li>
            <li>Development of <b>Time-Series Stress Response</b> modeling.</li>
            <li>Creation of a <b>Mobile App</b> for on-field stress detection.</li>
            <li>Implementation of a <b>REST API Endpoint</b> for external tool integration.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center;padding:15px;background-color:#E8F5E9;border-radius:10px;">
        <h3 style="color:#1B5E20;">🌾 RiML - Rice Machine Learning</h3>
        <p style="color:#388E3C;">Empowering rice stress research through data-driven insights</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background-color: #E8F5E9; border-radius: 10px;">
    <h3 style="color: #2E7D32;">🌾 RiML - Rice Machine Learning</h3>
    <p style="color: #666;">© 2025 | Powered by scikit-learn & Streamlit</p>
</div>
""", unsafe_allow_html=True)
