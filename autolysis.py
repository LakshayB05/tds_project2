# script
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "scipy",
#   "matplotlib",
#   "numpy",
#   "tabulate",
# ]


import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests  # Used for API calls 

# Load AIPROXY_TOKEN from environment variable
AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

# API endpoint for OpenAI proxy
url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def analyze_csv(filename):
    """
    Analyzes the dataset and generates insights through visualizations and AI-generated narrative.
    Outputs are saved in the current directory, not a subfolder.
    """
    try:
        # Attempt to load dataset with UTF-8 encoding
        data = pd.read_csv(filename, encoding='utf-8')
        print(f"Successfully loaded dataset: {filename}")
    except UnicodeDecodeError:
        # If UTF-8 fails, fall back to a common alternative encoding
        try:
            data = pd.read_csv(filename, encoding='ISO-8859-1')
            print(f"Successfully loaded dataset with ISO-8859-1 encoding: {filename}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

    # Extract dataset name (no subfolder creation)
    dataset_name = os.path.splitext(os.path.basename(filename))[0]

    # Generate basic summary statistics
    summary = data.describe(include='all').to_dict()
    missing_values = data.isnull().sum().to_dict()

    # Prepare for visualizations
    image_paths = []
    correlation_matrix = None

    # Check for numeric columns before proceeding
    numeric_cols = data.select_dtypes(include=['float64', 'int64'])
    
    if numeric_cols.empty:
        print("Warning: No numeric columns found in the dataset. Skipping correlation analysis, boxplot, and heatmaps.")
    else:
        # 1. Correlation Heatmap
        try:
            plt.figure(figsize=(5, 5))  # Set the size to 512x512 px (5x5 inches)
            correlation_matrix = numeric_cols.corr()  # Compute correlation matrix only for numeric columns
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title(f"Correlation Heatmap: {dataset_name}")
            correlation_image = f"{dataset_name}_correlation_heatmap.png"
            plt.savefig(correlation_image, dpi=150, bbox_inches="tight")
            image_paths.append(correlation_image)
            plt.close()
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")

        # 2. Missing Values Heatmap
        try:
            plt.figure(figsize=(5, 5))  # Ensure 512x512 px
            sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
            plt.title(f"Missing Values Heatmap: {dataset_name}")
            missing_values_image = f"{dataset_name}_missing_values_heatmap.png"
            plt.savefig(missing_values_image, dpi=150, bbox_inches="tight")
            image_paths.append(missing_values_image)
            plt.close()
        except Exception as e:
            print(f"Error generating missing values heatmap: {e}")

        # 3. Boxplot
        try:
            plt.figure(figsize=(5, 5))  # Ensure 512x512 px
            sns.boxplot(data=numeric_cols.dropna())
            plt.title(f"Boxplot: {dataset_name}")
            boxplot_image = f"{dataset_name}_boxplot.png"
            plt.savefig(boxplot_image, dpi=150, bbox_inches="tight")
            image_paths.append(boxplot_image)
            plt.close()
        except Exception as e:
            print(f"Error generating boxplot: {e}")

    try:
        # Optimized detailed prompt for AI generation
        prompt = f"""
        Below is the analysis summary for the dataset {dataset_name}:
        
        **Summary Statistics:** {summary}
        **Missing Values:** {missing_values}
        **Correlation Matrix:** {correlation_matrix.to_dict() if correlation_matrix is not None else 'N/A'}

        **Key Visual Insights:**
        1. **Correlation Heatmap**: Displays relationships between variables.
        2. **Missing Values Heatmap**: Highlights gaps in data.
        3. **Boxplot**: Identifies potential outliers and variable distributions.

        Write a business-focused report based on these insights:
        - Summarize key findings and any surprising trends.
        - Suggest specific actions or improvements based on patterns.
        - Explain how this data can guide strategic decisions or improvements.
        
        Please format the response in Markdown style for easy presentation.
        """

        # Request data for the API
        data_for_api = {
            "model": "gpt-4o-mini",  # Use the gpt-4o-mini model as required
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},  # Use the optimized detailed prompt
            ],
            "max_tokens": 1000  # Set the token limit
        }

        # Make the API call
        response = requests.post(url, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }, json=data_for_api)

        # Check the response status
        if response.status_code == 200:
            result = response.json()
            story = result['choices'][0]['message']['content'].strip()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"Error with LLM generation: {e}")
        sys.exit(1)
        

    # Save the story and visualizations in README.md
    try:
        readme_path = "README.md"
        with open(readme_path, "w") as f:
            f.write(f"# Analysis of {dataset_name}\n\n")
            f.write("## Dataset Insights and Recommendations\n\n")
            f.write("### Business Report\n")
            f.write(story)
            f.write("\n\n### Visualizations\n")
            for img_path in image_paths:
                f.write(f"![{os.path.basename(img_path)}]({os.path.basename(img_path)})\n")

        print(f"Analysis complete for {dataset_name}. Outputs saved in the current directory.")
    except Exception as e:
        print(f"Error saving README.md: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    analyze_csv(input_csv)
