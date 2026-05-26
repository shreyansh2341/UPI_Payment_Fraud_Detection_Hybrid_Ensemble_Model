import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_creditcard.csv")   

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

plt.figure(figsize=(5,4))
sns.countplot(x="isFraud", data=df, palette="Set2")
plt.title("Fraud vs Non-Fraud Count")
plt.show()

fraud_ratio = df['isFraud'].mean() * 100
print(f"Fraud percentage: {fraud_ratio:.4f}%")

plt.figure(figsize=(6,4))
sns.histplot(df['amount'], bins=100, log_scale=(True, True), kde=True, color="blue")
plt.title("Transaction Amount Distribution (log scale)")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="isFraud", y="amount", data=df, showfliers=False, palette="Set2")
plt.title("Transaction Amount by Fraud/Non-Fraud")
plt.ylim(0, df['amount'].quantile(0.95))  
plt.show()

if "hour" in df.columns:
    plt.figure(figsize=(7,4))
    sns.histplot(data=df, x="hour", hue="isFraud", multiple="stack", bins=24, palette="Set1")
    plt.title("Fraud Distribution by Hour of Day")
    plt.show()

plt.figure(figsize=(12,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap")
plt.show()
