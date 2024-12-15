# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai  # Ensure installation: pip install openai

# Function to load the dataset with flexible encoding
def load_dataset(file_path):
    print("Loading dataset with flexible encoding...")  # Debugging line
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        print(f"Dataset loaded successfully with UTF-8 encoding!")  # Debugging line
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, encoding='ISO-8859-1')
            print(f"Dataset loaded successfully with ISO-8859-1 encoding!")  # Debugging line
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    return data

# Function to analyze the dataset (summary stats, missing values, correlations)
def perform_data_analysis(data):
    print("Performing data analysis...")  # Debugging line
    summary = data.describe()
    missing = data.isnull().sum()
    
    # Ensure numeric conversion for correlation calculation
    numeric_data = data.select_dtypes(include=[np.number])
    for column in data.columns:
        if column not in numeric_data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
    numeric_data = data.select_dtypes(include=[np.number])

    correlations = numeric_data.corr() if not numeric_data.empty else pd.DataFrame()
    print("Data analysis completed.")  # Debugging line
    return summary, missing, correlations

# Function to detect anomalies using the IQR method
def identify_anomalies(data):
    print("Identifying anomalies...")  # Debugging line
    numeric_data = data.select_dtypes(include=[np.number])
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    anomalies = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()
    print("Anomaly identification completed.")  # Debugging line
    return anomalies

# Function to produce visualizations (heatmap, anomaly plot, distribution)
def generate_charts(corr_matrix, anomalies, data, save_dir):
    print("Creating visualizations...")  # Debugging line
    if not corr_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        heatmap_path = os.path.join(save_dir, 'correlation_matrix.png')
        plt.savefig(heatmap_path)
        plt.close()
    else:
        heatmap_path = None

    if not anomalies.empty and anomalies.sum() > 0:
        plt.figure(figsize=(10, 6))
        anomalies.plot(kind='bar', color='red')
        plt.title('Anomaly Detection')
        anomaly_plot_path = os.path.join(save_dir, 'anomalies.png')
        plt.savefig(anomaly_plot_path)
        plt.close()
    else:
        anomaly_plot_path = None

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if numeric_cols.size > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[numeric_cols[0]], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {numeric_cols[0]}')
        distribution_path = os.path.join(save_dir, 'distribution_plot.png')
        plt.savefig(distribution_path)
        plt.close()
    else:
        distribution_path = None

    print("Visualizations created.")  # Debugging line
    return heatmap_path, anomaly_plot_path, distribution_path

# Function to craft the README.md report
def compile_report(summary, missing, corr_matrix, anomalies, charts_dir):
    print("Compiling report...")  # Debugging line
    readme_path = os.path.join(charts_dir, 'README.md')
    try:
        with open(readme_path, 'w') as file:
            file.write("# Data Analysis Report\n\n")
            file.write("## Summary\nThis report provides a detailed analysis of the dataset, exploring its structure, anomalies, and inter-variable relationships.\n\n")
            
            file.write("### Summary Statistics\n")
            file.write(summary.to_markdown() + "\n\n")

            file.write("### Missing Values\n")
            file.write(missing.to_markdown() + "\n\n")

            file.write("### Correlation Matrix\n")
            if not corr_matrix.empty:
                file.write("![Correlation Matrix](correlation_matrix.png)\n\n")
            else:
                file.write("No correlations available for non-numeric data.\n\n")

            file.write("### Anomalies\n")
            if anomalies.sum() > 0:
                file.write("![Anomalies](anomalies.png)\n\n")
            else:
                file.write("No significant anomalies detected.\n\n")

            file.write("### Distribution\n")
            file.write("![Distribution Plot](distribution_plot.png)\n\n")

        print(f"Report compiled: {readme_path}")  # Debugging line
        return readme_path
    except Exception as e:
        print(f"Error creating report: {e}")
        return None

# Function to invoke an LLM for a narrative
def generate_narrative_via_llm(prompt, details):
    print("Calling LLM for narrative generation...")  # Debugging line
    token = os.getenv("AIPROXY_TOKEN")
    if not token:
        print("Error: AIPROXY_TOKEN not set.")
        return "Unable to generate narrative."

    try:
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}\n\nContext:\n{details}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        response = requests.post(api_url, headers={"Authorization": f"Bearer {token}"}, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"LLM request error: {response.status_code}")
            return "Narrative generation failed."
    except Exception as e:
        print(f"Error: {e}")
        return "Error during narrative generation."

# Main function
def main(file_path):
    print("Starting data analysis process...")  # Debugging line
    try:
        data = load_dataset(file_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    summary, missing, corr_matrix = perform_data_analysis(data)
    anomalies = identify_anomalies(data)

    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    heatmap, anomalies_chart, distribution = generate_charts(corr_matrix, anomalies, data, output_dir)

    narrative = generate_narrative_via_llm(
        "Write a detailed analysis story based on the dataset.", 
        details=f"Summary: {summary}\nMissing: {missing}\nAnomalies: {anomalies}"
    )

    report_path = compile_report(summary, missing, corr_matrix, anomalies, output_dir)
    if report_path:
        with open(report_path, 'a') as report:
            report.write("## Narrative\n")
            report.write(narrative)

    print("Data analysis completed successfully!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
