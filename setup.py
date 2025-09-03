from setuptools import setup, find_packages

setup(
    name="drug_response_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "biopython",
        "gseapy",
        "jupyter",
        "lifelines",
        "matplotlib",
        "numpy",
        "pandas",
        "plotly",
        "scikit-learn",
        "seaborn",
        "statsmodels",
        "streamlit",
        "requests",
    ],
    python_requires=">=3.8",
)