# RiML--Rice-stress-tolerance-predictor
# 🌾 RiML - Rice Machine Learning

**RiML** is a web-based machine learning application for classifying rice stress conditions from gene expression data. Built with Streamlit and powered by a Random Forest classifier, RiML helps researchers identify drought, salt, cold, and control conditions in rice samples.

---

## 🎯 Features

- **4 Stress Type Classification**: Control, Drought, Salt, and Cold stress detection
- **High Accuracy**: 91.7% cross-validation accuracy with 200 stress-responsive genes
- **User-Friendly Interface**: Easy-to-use web interface with drag-and-drop file upload
- **Interactive Visualizations**: Publication-ready pie charts, bar charts, and probability heatmaps
- **GPL Annotation Support**: Optional gene annotation file upload for enriched biological insights
- **Robust Error Handling**: Comprehensive data validation with helpful error messages
- **Downloadable Results**: Export predictions and gene annotations as CSV files
- **Real-time Analysis**: Fast processing with immediate feedback and progress indicators

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Classifier |
| Trees | 100 |
| Features | 200 stress-responsive genes |
| CV Accuracy | 91.7% |
| Validation | Leave-One-Out Cross-Validation |
| Dataset | GSE6901 (Gene Expression Omnibus) |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

```bash
git clone <repository-url>
cd riml
```

### Step 2: Install Dependencies

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 3: Required Files

Make sure these files are in the same directory as `app.py`:

```
riml/
├── app.py
├── Pretrained_rice_model.pkl
├── SCALER.pkl
├── Top_genes.pkl
├── MODEL_info.pkl
└── README.md
```

---

## 🏃 Running the Application

### Start the App

```bash
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`

### Alternative: Run with Custom Port

```bash
streamlit run app.py --server.port 8080
```

---

## 📖 How to Use

### Step 1: Navigate to Analysis Tab

Click on the **🔬 Analysis** tab in the application.

### Step 2: Upload Gene Expression File (Required)

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
- First column: Gene IDs (ID_REF, Gene, gene_id, etc.)
- Other columns: Sample expression values (numeric)
- Rows: Genes
- Columns: Samples

### Step 3: Upload GPL Annotation File (Optional)

Upload a GPL platform annotation file to get:
- Gene symbols
- Gene descriptions
- Enhanced interpretation of results

**Example:** `GPL2025-2968.txt`

### Step 4: Click ANALYZE

Click the **🚀 ANALYZE DATA** button to start the analysis.

### Step 5: View Results

- **Summary Metrics**: Sample count, confidence, stress types
- **Detailed Predictions Table**: Per-sample predictions with probabilities
- **Visualizations**: Distribution charts and heatmaps
- **Gene Information**: Annotations (if GPL file uploaded)

### Step 6: Download Results

Download your predictions and gene annotations as CSV files.

---

## 📁 Input File Examples

### Gene Expression File (CSV)

```csv
ID_REF,Control_1,Control_2,Drought_1,Salt_1
Os01g0100100,245.5,312.8,198.3,267.4
Os01g0100200,567.2,523.9,601.5,489.3
Os02g0100300,123.7,145.2,118.9,156.8
```

### GPL Annotation File (TSV)

```tsv
ID	Gene_Symbol	Description
Os01g0100100	OsACT1	Actin protein - structural component
Os01g0100200	OsWRKY1	WRKY transcription factor 1
Os02g0100300	OsDREB1	Dehydration-responsive element-binding protein
```

---

## 🎨 Application Structure

### Tab 1: Home 🏠

- Overview of the application
- Model performance metrics
- Quick start guide
- Stress types information

### Tab 2: Analysis 🔬

- File upload interface
- Data validation
- Prediction engine
- Results visualization
- Download options

### Tab 3: About ℹ️

- Model details
- Acknowledgments
- Future development plans
- Team information

---


---

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
```

### Create requirements.txt

```bash
echo "streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0" > requirements.txt
```

---

## 🌾 Stress Types Detected

| Icon | Stress Type | Description |
|------|-------------|-------------|
| 🌱 | **Control** | Normal, unstressed conditions |
| 🏜️ | **Drought** | Water deficit stress |
| 🧂 | **Salt** | High salinity stress |
| ❄️ | **Cold** | Low temperature stress |

---

## 🔬 Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Gene ID validation
   - Missing value imputation
   - AFFX control probe removal
   - Data normalization

2. **Feature Selection**
   - 200 stress-responsive genes
   - Selected based on importance scores
   - Validated across multiple stress types

3. **Model Training**
   - Random Forest with 100 trees
   - Maximum depth of 5
   - Leave-One-Out Cross-Validation

4. **Prediction**
   - Probability scores for each stress type
   - Confidence-based classification

### Data Flow

```
Upload File → Validate → Preprocess → Scale → Predict → Visualize → Download
```

---

## 📊 Example Output

### Predictions Table

| Sample | Stress | Confidence | Control % | Drought % | Salt % | Cold % |
|--------|--------|------------|-----------|-----------|--------|--------|
| Sample1 | 🌱 Control | 95.2% | 95.2 | 2.3 | 1.5 | 1.0 |
| Sample2 | 🏜️ Drought | 88.7% | 5.1 | 88.7 | 3.2 | 3.0 |
| Sample3 | 🧂 Salt | 92.4% | 2.8 | 3.1 | 92.4 | 1.7 |

---

## 🚀 Future Work

- [ ] Support for more stress types such as **Heat** and **UV**
- [ ] Integration of **RNA-Seq** datasets for improved analysis
- [ ] Development of **Time-Series Stress Response** modeling
- [ ] Creation of a **Mobile App** for on-field stress detection
- [ ] Implementation of a **REST API Endpoint** for external tool integration
- [ ] Batch processing capabilities for large-scale datasets
- [ ] Advanced visualization options and custom reports
- [ ] Model retraining interface for user-specific datasets

---

## 🙏 Acknowledgments

- **Mentor:** Dr. Kushagrah Kashyap - For invaluable guidance, continuous support, and insightful feedback throughout the development of RiML
- **Data Source**: GEO (Gene Expression Omnibus) - GSE6901
- **Platform**: Affymetrix Rice Genome Array
- **Tools**: scikit-learn, Streamlit, Plotly
- **Community**: Rice research scientists worldwide

---

## 👥 Team

**Developers:** Pranjal Dalvi and Sejal Chaudhari  
**Institution:** DES Pune University  
**Contact:** dpranjal889@gmail.com  
**LinkedIn:** [Pranjal Dalvi](https://www.linkedin.com/in/pranjal-dalvi-2b6538316)

---



---

## 📞 Support

For questions, issues, or contributions:

- **Email:** dpranjal889@gmail.com
- **LinkedIn:** [Pranjal Dalvi](https://www.linkedin.com/in/pranjal-dalvi-2b6538316)
- **Institution:** DES Pune University

---


---


---

**Made with ❤️ for rice research**

🌾 **RiML - Empowering Rice Stress Analysis with Machine Learning** 🌾
