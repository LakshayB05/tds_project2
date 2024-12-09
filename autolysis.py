import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import openai
import uvicorn

openai.api_key = os.environ["AIPROXY_TOKEN"]  # Use the environment variable for API token

def load_data(file_path):
    return pd.read_csv(file_path)

def analyze_data(df):
    # Example of generic analysis
    summary = df.describe()  # Summary statistics
    missing_values = df.isnull().sum()  # Missing values
    correlation_matrix = df.corr()  # Correlation matrix
    return summary, missing_values, correlation_matrix

def visualize_data(df, correlation_matrix):
    # Example of creating a heatmap for correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.savefig("correlation_heatmap.png")  # Save as PNG

def generate_story(summary, missing_values, correlation_matrix):
    prompt = f"Data summary:\n{summary}\n\nMissing values:\n{missing_values}\n\nCorrelation matrix:\n{correlation_matrix}\n\nNarrate a story from the data with insights and implications."
    response = openai.Completion.create(engine="gpt-4o-mini", prompt=prompt, max_tokens= 1000)
    return response['choices'][0]['text']

def save_output(story, output_dir):
    # Save the README.md and images in the output directory
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(story)
    plt.savefig(f"{output_dir}/correlation_heatmap.png")

if __name__ == "__main__":
    file_path = "your_dataset.csv"  # Change this to your dataset path
    df = load_data(file_path)
    summary, missing_values, correlation_matrix = analyze_data(df)
    visualize_data(df, correlation_matrix)
    story = generate_story(summary, missing_values, correlation_matrix)
    save_output(story, "output")
