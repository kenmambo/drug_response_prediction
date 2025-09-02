import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from Bio import Entrez
import gseapy as gp

# Load model and preprocessing objects
model = joblib.load('best_model.pkl')
expression_scaler = joblib.load('expression_scaler.pkl')
feature_selector = joblib.load('feature_selector.pkl')
selected_genes = joblib.load('selected_genes.pkl')

# App title
st.title('Drug Response Prediction Dashboard')

# Sidebar for inputs
st.sidebar.header('Genomic Data Input')

def load_example_data():
    """Load example genomic data"""
    # In practice, this would come from a genomic database
    # Here we'll generate synthetic data for demonstration
    np.random.seed(42)
    example_data = {}
    for gene in selected_genes[:20]:  # Show top 20 genes
        example_data[gene] = np.random.normal(0, 1)
    return example_data

def user_input_features():
    """Create input widgets for genomic data"""
    st.sidebar.write("Enter gene expression values (z-scores):")
    
    # Load example data
    example_data = load_example_data()
    
    # Create sliders for top 20 genes
    genomic_data = {}
    for gene in selected_genes[:20]:
        genomic_data[gene] = st.sidebar.slider(
            f'{gene}', 
            min_value=-3.0, 
            max_value=3.0, 
            value=example_data[gene],
            step=0.1
        )
    
    # For remaining genes, use example values
    for gene in selected_genes[20:]:
        genomic_data[gene] = example_data.get(gene, 0.0)
    
    return pd.DataFrame([genomic_data])

input_df = user_input_features()

# Display input data
st.subheader('Genomic Profile')
st.write(f"Input data for {len(selected_genes)} genes")

# Make prediction
if st.button('Predict Drug Response'):
    # Preprocess input
    input_processed = feature_selector.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_processed)[0]
    
    # Convert back from log scale
    ic50_prediction = 10 ** prediction
    
    # Display prediction
    st.subheader('Drug Response Prediction')
    st.metric("Predicted IC50", f"{ic50_prediction:.2f} μM")
    st.write(f"Log IC50: {prediction:.2f}")
    
    # Interpretation
    st.subheader('Interpretation')
    if ic50_prediction < 1:
        st.success("High sensitivity to the drug")
        st.write("The patient is likely to respond well to 5-Fluorouracil treatment.")
    elif ic50_prediction < 10:
        st.warning("Moderate sensitivity to the drug")
        st.write("The patient may respond to 5-Fluorouracil, but combination therapy might be considered.")
    else:
        st.error("Low sensitivity to the drug")
        st.write("The patient is unlikely to respond well to 5-Fluorouracil. Alternative treatments should be considered.")

# Feature importance
st.subheader('Feature Importance')
if hasattr(model, 'feature_importances_'):
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Gene': [selected_genes[i] for i in indices],
        'Importance': importances[indices]
    })
    
    # Plot
    fig = px.bar(importance_df, x='Importance', y='Gene', orientation='h',
                 title='Top 15 Important Genes for Drug Response')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# Pathway enrichment
st.subheader('Pathway Enrichment Analysis')
if st.button('Run Pathway Analysis'):
    with st.spinner('Analyzing enriched pathways...'):
        # Get top genes
        top_genes = selected_genes[:100]  # Use top 100 genes
        
        # Run enrichment analysis
        enr_results = gp.enrichr(
            gene_list=top_genes.tolist(),
            gene_sets=['KEGG_2021_Human'],
            organism='human',
            outdir=None
        )
        
        # Display results
        st.dataframe(enr_results.results.head(10)[['Term', 'P-value', 'Adjusted P-value', 'Odds Ratio']])
        
        # Create bar plot
        top_pathways = enr_results.results.head(10)
        fig = px.bar(
            top_pathways, 
            x='Odds Ratio', 
            y='Term',
            color='Adjusted P-value',
            title='Top Enriched Pathways',
            labels={'Odds Ratio': 'Odds Ratio', 'Term': 'Pathway'},
            orientation='h'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# Model performance
st.subheader('Model Performance')
results_df = pd.read_csv('model_results.csv', index_col=0)
st.dataframe(results_df)

# Create performance comparison chart
fig = go.Figure()
fig.add_trace(go.Bar(
    x=results_df.index,
    y=results_df['R2'],
    name='R² Score',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=results_df.index,
    y=results_df['RMSE'],
    name='RMSE',
    marker_color='lightsalmon',
    yaxis='y2'
))

fig.update_layout(
    title='Model Performance Comparison',
    xaxis_title='Model',
    yaxis_title='R² Score',
    yaxis2=dict(
        title='RMSE',
        overlaying='y',
        side='right'
    ),
    barmode='group'
)
st.plotly_chart(fig, use_container_width=True)