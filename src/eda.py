import matplotlib.pyplot as plt
import seaborn as sns


def correlation_heatmap(df):

    plt.figure(figsize=(18,12))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()



def price_distribution(df):

    plt.figure(figsize=(8,5))
    sns.histplot(df['price'], kde=True)
    plt.title('Price Distribution')
    plt.show()