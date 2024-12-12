# /// script
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "scipy",
#   "matplotlib",
#   "numpy",
#   "chardet",
#   "tabulate",
#   "requests",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import chardet
from scipy.stats import skew, kurtosis

# Load AIPROXY_TOKEN from environment variable
AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

# API endpoint for OpenAI proxy
url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def detect_encoding(file_path):
    """
    Detect file encoding to handle diverse datasets gracefully.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def calculate_advanced_statistics(data):
    """
    Calculate advanced statistics such as skewness and kurtosis for numeric columns.
    """
    numeric_cols = data.select_dtypes(include=['float64', 'int64'])
    if numeric_cols.empty:
        return "No numeric columns found."
    return {
        "Skewness": numeric_cols.apply(skew).to_dict(),
        "Kurtosis": numeric_cols.apply(kurtosis).to_dict(),
    }

def create_visualizations(data, dataset_name):
    """
    Generate key visualizations and return paths to the saved images.
    """
    image_paths = []
    numeric_cols = data.select_dtypes(include=['float64', 'int64'])

    # Correlation Heatmap
    if not numeric_cols.empty:
        try:
            plt.figure(figsize=(5, 5))
            correlation_matrix = numeric_cols.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title(f"Correlation Heatmap: {dataset_name}")
            plt.xlabel("Features")
            plt.ylabel("Features")
            correlation_image = f"{dataset_name}_correlation_heatmap.png"
            plt.savefig(correlation_image, dpi=150, bbox_inches="tight")
            image_paths.append(correlation_image)
            plt.close()
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")

    # Missing Values Heatmap
    try:
        plt.figure(figsize=(5, 5))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
        plt.title(f"Missing Values Heatmap: {dataset_name}")
        missing_values_image = f"{dataset_name}_missing_values_heatmap.png"
        plt.savefig(missing_values_image, dpi=150, bbox_inches="tight")
        image_paths.append(missing_values_image)
        plt.close()
    except Exception as e:
        print(f"Error generating missing values heatmap: {e}")

    return image_paths

def generate_prompt(data_summary, stats, correlation_matrix, dataset_name):
    """
    Generate a context-rich prompt for the LLM.
    """
    return f"""
    Below is the analysis summary for the dataset {dataset_name}:

    **Summary Statistics:** {data_summary}
    **Advanced Statistics:** {stats}
    **Correlation Matrix:** {correlation_matrix}

    Key Insights:
    - Describe relationships, gaps, and trends in the dataset.
    - Recommend actions based on findings.
    - Highlight strategic implications of these insights.

    Please provide a business-focused report in Markdown format.
    """

def call_llm(prompt):
    """
    Make an API call to the LLM with the generated prompt.
    """
    data_for_api = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000
    }
    response = requests.post(url, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }, json=data_for_api)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        sys.exit(1)

def analyze_csv(filename):
    """
    Main function to analyze the dataset and integrate LLM for insights.
    """
    # Load dataset
    encoding = detect_encoding(filename)
    data = pd.read_csv(filename, encoding=encoding)
    dataset_name = os.path.splitext(os.path.basename(filename))[0]

    # Data summaries and statistics
    data_summary = data.describe(include='all').to_dict()
    missing_values = data.isnull().sum().to_dict()
    advanced_stats = calculate_advanced_statistics(data)
    correlation_matrix = data.corr().to_dict() if not data.empty else "N/A"

    # Create visualizations
    image_paths = create_visualizations(data, dataset_name)
    selected_image = image_paths[0] if image_paths else None

    # Generate prompt and call LLM
    prompt = generate_prompt(data_summary, advanced_stats, correlation_matrix, dataset_name)
    narrative = call_llm(prompt)

    # Save results to README
    with open("README.md", "w") as f:
        f.write(f"# Analysis of {dataset_name}\n\n")
        f.write("## Insights and Recommendations\n\n")
        f.write("### Business Report\n")
        f.write(narrative)
        f.write("\n\n### Visualizations\n")
        for img_path in image_paths:
            f.write(f"![{os.path.basename(img_path)}]({os.path.basename(img_path)})\n")

    print(f"Analysis complete for {dataset_name}. Results saved in README.md.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)
    analyze_csv(sys.argv[1])
