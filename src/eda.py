import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df):

    os.makedirs('plots', exist_ok=True)

    # Charges Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['charges'], kde=True)
    plt.title('Charges Distribution')
    plt.savefig('plots/charges_distribution.png')
    plt.close()

    # BMI Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['bmi'], kde=True)
    plt.title('BMI Distribution')
    plt.savefig('plots/bmi_distribution.png')
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()

    # Age vs Charges
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df['age'], y=df['charges'])
    plt.title('Age vs Charges')
    plt.savefig('plots/age_vs_charges.png')
    plt.close()