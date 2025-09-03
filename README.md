# 🧬 Drug Response Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project aims to predict cancer cell line responses to drug treatments using genomic data. By leveraging machine learning models trained on gene expression profiles, we can predict how effective specific drugs will be against different cancer types, potentially accelerating personalized cancer treatment.

<div align="center">

```
Genomic Data → ML Models → Drug Response Prediction
```

</div>

## ✨ Features

- 🔬 **Genomic Data Processing**: Preprocess gene expression data from cancer cell lines
- 🧪 **Drug Response Analysis**: Analyze and predict IC50 values for drug effectiveness
- 🧠 **Multiple ML Models**: Compare Random Forest, Gradient Boosting, ElasticNet, and SVR models
- 📊 **Interactive Dashboard**: Visualize predictions and model performance with Streamlit
- 🔍 **Pathway Enrichment**: Identify significant biological pathways in selected genes
- 📈 **Model Evaluation**: Comprehensive metrics including RMSE, MAE, and R² scores

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or uv package manager

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/drug-response-prediction.git
   cd drug-response-prediction
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies
   ```bash
   uv pip install -e .
   ```

### Data Preparation

Run the data preparation script to download required datasets or create mock data:

```bash
python download_and_prepare_data.py
```

## 🔄 Workflow

1. **Data Preprocessing**
   - Load cell line, gene expression, and drug response data
   - Normalize gene expression values
   - Filter genes by variance
   - Select relevant features

2. **Model Training**
   - Train multiple regression models
   - Perform hyperparameter tuning
   - Evaluate model performance
   - Select best performing model

3. **Prediction & Visualization**
   - Use trained model to predict drug responses
   - Visualize results through interactive dashboard
   - Explore feature importance and pathway enrichment

## 📊 Dashboard

Launch the interactive dashboard:

```bash
streamlit run app/dashboard.py
```

The dashboard allows you to:
- Input custom gene expression values
- Visualize predicted drug responses
- Explore model performance metrics
- Analyze feature importance

## 🧪 Running the Pipeline

To run the complete analysis pipeline:

```bash
python run_pipeline.py
```

This will:
1. Load and preprocess the data
2. Perform pathway enrichment analysis
3. Train and evaluate models
4. Save the best model and preprocessing objects

## 📁 Project Structure

```
├── app/
│   └── dashboard.py         # Streamlit dashboard
├── data/                    # Data directory (created by download script)
├── notebooks/              
│   ├── EDA.ipynb            # Exploratory data analysis
│   └── download_data.py     # Data download script
├── src/
│   ├── data_preprocessing.py # Data preprocessing functions
│   └── model_training.py     # Model training functions
├── download_and_prepare_data.py # Data preparation script
├── run_pipeline.py          # Main pipeline script
├── main.py                  # Entry point
└── README.md                # This file
```

## 🔮 Future Enhancements

### Near-Term Goals
- **Multi-Drug Prediction**: Extend to predict response for multiple drugs simultaneously
- **Clinical Data Integration**: Incorporate patient clinical variables with genomic data
- **Survival Analysis**: Predict not just response but also progression-free survival

### Long-Term Vision
- **Single-Cell Resolution**: Apply to single-cell RNA-seq data for tumor heterogeneity
- **Drug Combination Prediction**: Model synergistic effects of drug combinations
- **Real-Time Clinical Decision Support**: Integrate with hospital EMR systems

## 📚 References

- [Cancer Cell Line Encyclopedia (CCLE)](https://sites.broadinstitute.org/ccle/)
- [Genomics of Drug Sensitivity in Cancer (GDSC)](https://www.cancerrxgene.org/)
- [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/tcga)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Kenneth Mambo - *Initial work*

---

<div align="center">

**Made with ❤️ for advancing cancer research**

</div>