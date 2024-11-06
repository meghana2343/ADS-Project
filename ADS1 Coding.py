# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

# Step 1: Load and Clean Data
def load_and_clean_data(file_path):
    """
    Loads and cleans the Billionaires dataset, handling missing values.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for analysis.
    """
    data = pd.read_csv(file_path)
    
    # Fill missing values with median for numeric and mode for categorical columns
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            data[column].fillna(data[column].median(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

# Load and clean the dataset
file_path = 'Billionaires Statistics Dataset.csv'
billionaires_data = load_and_clean_data(file_path)

# Step 2: Statistical Analysis
def display_statistics(data):
    """
    Displays summary statistics, correlations, skewness, and kurtosis.

    Args:
        data (pd.DataFrame): The cleaned dataset.

    Returns:
        None
    """
    print("Summary Statistics:\n", data.describe())

    # Select numeric columns for correlation
    numeric_data = data.select_dtypes(include=['number'])

    # Correlation matrix
    print("\nCorrelation Matrix:\n", numeric_data.corr())

    # Skewness and Kurtosis
    skewness = numeric_data.apply(lambda x: skew(x.dropna()))
    kurt = numeric_data.apply(lambda x: kurtosis(x.dropna()))
    print("\nSkewness:\n", skewness)
    print("\nKurtosis:\n", kurt)

# Display statistical information
display_statistics(billionaires_data)

# Step 3: Visualization Functions

# Plot 1: Simplified Pie Chart - Top 10 Billionaire Industries
def plot_top_10_industries(data):
    """
    Plots a pie chart showing the proportion of billionaires in the top 10 industries,
    grouping all other industries into an "Other" category.
    """
    if 'category' in data.columns:
        # Count of billionaires by category and filter out small categories
        category_counts = data['category'].value_counts()
        
        # Keep only the top 10 categories, group the rest as 'Other'
        top_categories = category_counts.nlargest(10)
        other_count = category_counts[10:].sum()
        top_categories['Other'] = other_count
        
        # Plot pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(top_categories, labels=top_categories.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Billionaires by Top 10 Industry Categories', fontsize=14, fontweight='bold')
        plt.show()
    else:
        print("Column 'category' not found in dataset for pie chart.")

# Plot 2: Simplified Box Plot - Net Worth Distribution by Top 10 Countries
def plot_net_worth_by_country(data):
    """
    Plots a box plot comparing the distribution of net worth for the top 10 countries
    with the highest total net worth.
    """
    if 'country' in data.columns and 'finalWorth' in data.columns:
        # Calculate total net worth by country and select top 10 countries
        top_countries = data.groupby('country')['finalWorth'].sum().nlargest(10).index
        filtered_data = data[data['country'].isin(top_countries)]
        
        # Plot box plot
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='country', y='finalWorth', data=filtered_data)
        plt.title('Net Worth of Billionaires by Top 10 Countries', fontsize=14, fontweight='bold')
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Net Worth (in billions)', fontsize=12)
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("Columns 'country' or 'finalWorth' not found in dataset for box plot.")

# Plot 3: Bar Chart - Top Countries by Number of Billionaires (with custom colors)
def plot_top_billionaire_countries(data, top_n=10):
    """
    Plots a bar chart showing the top countries by the number of billionaires with custom colors.

    Args:
        data (pd.DataFrame): The dataset containing country information.
        top_n (int): The number of top countries to display (default is 10).
    """
    if 'country' in data.columns:
        # Count the number of billionaires by country and select top countries
        country_counts = data['country'].value_counts().nlargest(top_n)
        
        # Plot bar chart with custom colors
        plt.figure(figsize=(10, 6))
        sns.barplot(x=country_counts.values, y=country_counts.index, palette="viridis")
        plt.title(f'Top {top_n} Countries by Number of Billionaires', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Billionaires', fontsize=12)
        plt.ylabel('Country', fontsize=12)
        plt.show()
    else:
        print("Column 'country' not found in dataset for bar chart.")

# Scatter Plot - Net Worth vs. Age of Billionaires (Top 5 Industries)
def plot_net_worth_vs_age_filtered(data):
    """
    Plots a scatter plot showing the relationship between net worth and age of billionaires,
    filtered to show only the top 5 industries with the most billionaires.
    """
    if 'age' in data.columns and 'finalWorth' in data.columns and 'category' in data.columns:
        # Select top 5 industries by count of billionaires
        top_industries = data['category'].value_counts().nlargest(5).index
        filtered_data = data[data['category'].isin(top_industries)]
        
        # Plot scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='age', 
            y='finalWorth', 
            data=filtered_data, 
            hue='category', 
            palette='Set1', 
            s=70,  # Increased size for better visibility
            alpha=0.6  # Reduced transparency for clearer overlaps
        )
        plt.title('Net Worth vs Age of Billionaires (Top 5 Industries)', fontsize=14, fontweight='bold')
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Net Worth (in billions)', fontsize=12)
        plt.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
    else:
        print("Columns 'age', 'finalWorth', or 'category' not found in dataset for scatter plot.")

# Generate the plots
plot_top_10_industries(billionaires_data)
plot_net_worth_by_country(billionaires_data)
plot_top_billionaire_countries(billionaires_data, top_n=10)
plot_net_worth_vs_age_filtered(billionaires_data)