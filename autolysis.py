import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json

# Function to analyze the data (basic summary stats, missing values, correlation matrix)
def analyze_data(df):
    print("Analyzing the data...")  # Debugging line
    # Summary statistics for numerical columns
    summary_stats = df.describe()

    # Check for missing values
    missing_values = df.isnull().sum()

    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])

    # Correlation matrix for numerical columns
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()

    print("Data analysis complete.")  # Debugging line
    return summary_stats, missing_values, corr_matrix

# Function to detect outliers using the IQR method
def detect_outliers(df):
    print("Detecting outliers...")  # Debugging line
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Apply the IQR method to find outliers in the numeric columns
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()

    print("Outliers detection complete.")  # Debugging line
    return outliers

# Function to generate visualizations (correlation heatmap, outliers plot, and distribution plot)
def visualize_data(corr_matrix, outliers, df, output_dir):
    print("Generating visualizations...")  # Debugging line
    # Generate a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(heatmap_file)
    plt.close()

    # Check if there are outliers to plot
    if not outliers.empty and outliers.sum() > 0:
        # Plot the outliers
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_file = os.path.join(output_dir, 'outliers.png')
        plt.savefig(outliers_file)
        plt.close()
    else:
        print("No outliers detected to visualize.")
        outliers_file = None  # No file created for outliers

    # Generate a distribution plot for the first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_numeric_column = numeric_columns[0]  # Get the first numeric column
        plt.figure(figsize=(10, 6))
        sns.histplot(df[first_numeric_column], kde=True, color='blue', bins=30)
        plt.title(f'Distribution')
        dist_plot_file = os.path.join(output_dir, f'distribution_.png')
        plt.savefig(dist_plot_file)
        plt.close()
    else:
        dist_plot_file = None  # No numeric columns to plot

    print("Visualizations generated.")  # Debugging line
    return heatmap_file, outliers_file, dist_plot_file

# Function to generate a detailed story using the new OpenAI API through the proxy
def question_llm(prompt, context, max_tokens=1000):
    print("Generating story using LLM...")  # Debugging line
    try:
        # Get the AIPROXY_TOKEN from the environment variable
        token = os.environ["AIPROXY_TOKEN"]

        # Set the custom API base URL for the proxy
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        # Construct the full prompt
        full_prompt = f"""
        Based on the following data analysis, please generate a creative and engaging story. The story should include multiple paragraphs, a clear structure with an introduction, body, and conclusion, and should feel like a well-rounded narrative.

        Context:
        {context}

        Data Analysis Prompt:
        {prompt}

        The story should be elaborate and cover the following:
        - An introduction to set the context.
        - A detailed body that expands on the data points and explores their significance.
        - A conclusion that wraps up the analysis and presents any potential outcomes or lessons.
        - Use transitions to connect ideas and keep the narrative flowing smoothly.
        - Format the story with clear paragraphs and structure.
        """

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        # Prepare the body with the model and prompt
        data = {
            "model": "gpt-4o-mini",  # Specific model for proxy
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        # Send the POST request to the proxy
        response = requests.post(api_url, headers=headers, data=json.dumps(data))

        # Check for successful response
        if response.status_code == 200:
            # Extract the story from the response
            story = response.json()['choices'][0]['message']['content'].strip()
            print("Story generated.")  # Debugging line
            return story
        else:
            print(f"Error with request: {response.status_code} - {response.text}")
            return "Failed to generate story."

    except Exception as e:
        print(f"Error: {e}")
        return "Failed to generate story."

# Main function that integrates all the steps
def main(csv_file):
    print("Starting the analysis...")  # Debugging line

    # Try reading the CSV file with 'ISO-8859-1' encoding to handle special characters
    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully!")  # Debugging line
    except UnicodeDecodeError as e:
        print(f"Error reading file: {e}")
        return

    summary_stats, missing_values, corr_matrix = analyze_data(df)
    outliers = detect_outliers(df)

    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    # Visualize the data
    heatmap_file, outliers_file, dist_plot_file = visualize_data(corr_matrix, outliers, df, output_dir)

    # Limit the context size for the LLM to ensure it fits within 1000 tokens
    context_summary = f"""
    Dataset Analysis:
    Summary Statistics (First 5 columns):
    {summary_stats.iloc[:, :5]}

    Missing Values:
    {missing_values.head()}

    Correlation Matrix (First 5 columns):
    {corr_matrix.iloc[:, :5]}

    Outliers (First 5 columns):
    {outliers.head()}
    """

    # Generate the story using the LLM
    story = question_llm("Generate a nice and creative story from the analysis", 
                         context=context_summary, 
                         max_tokens=1000)

    print(f"Generated Story:\n{story}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
