import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# -----------------------------
# LOAD DATA (NO BROKEN SRC IMPORTS)
# -----------------------------
DATA_PATH = os.path.join("data", "raw", "data.csv")
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print(df.columns)


# -----------------------------
# SAFETY CLEANUP (IMPORTANT)
# -----------------------------
df.columns = df.columns.str.strip()


# -----------------------------
# TRY TO FIX COMMON XENTE COLUMN ISSUES
# -----------------------------
# Standardize column names if needed
rename_map = {
    "Transaction Start Time": "TransactionStartTime",
    "transactionstarttime": "TransactionStartTime",
    "amount": "Amount",
}

df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)


# -----------------------------
# CREATE FIGURES FOLDER
# -----------------------------
os.makedirs("figures", exist_ok=True)


# =========================================================
# FIGURE 1: MISSING VALUES
# =========================================================
plt.figure(figsize=(10,5))
df.isnull().mean().sort_values(ascending=False).head(15).plot(kind="bar")
plt.title("Figure 1: Missing Values Rate")
plt.tight_layout()
plt.savefig("figures/fig1_missing_values.png")
plt.show()


# =========================================================
# FIGURE 2: DATA TYPES
# =========================================================
plt.figure(figsize=(6,4))
df.dtypes.value_counts().plot(kind="bar")
plt.title("Figure 2: Data Types Distribution")
plt.tight_layout()
plt.savefig("figures/fig2_dtypes.png")
plt.show()


# =========================================================
# FIGURE 3: BOX PLOT (AMOUNT)
# =========================================================
if "Amount" in df.columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df["Amount"])
    plt.title("Figure 3: Boxplot of Transaction Amount")
    plt.tight_layout()
    plt.savefig("figures/fig3_boxplot_amount.png")
    plt.show()
else:
    print("WARNING: Amount column not found")


# =========================================================
# FIGURE 4: HISTOGRAM (AMOUNT)
# =========================================================
if "Amount" in df.columns:
    plt.figure(figsize=(8,4))
    plt.hist(df["Amount"].dropna(), bins=50)
    plt.title("Figure 4: Transaction Amount Distribution")
    plt.tight_layout()
    plt.savefig("figures/fig4_hist_amount.png")
    plt.show()


# =========================================================
# FIGURE 5: LOG DISTRIBUTION
# =========================================================
if "Amount" in df.columns:
    plt.figure(figsize=(8,4))
    plt.hist(df["Amount"].clip(lower=1), bins=50)
    plt.yscale("log")
    plt.title("Figure 5: Log Distribution of Amount")
    plt.tight_layout()
    plt.savefig("figures/fig5_log_amount.png")
    plt.show()


# =========================================================
# FIGURE 6: PRODUCT CATEGORY
# =========================================================
if "ProductCategory" in df.columns:
    plt.figure(figsize=(10,4))
    df["ProductCategory"].value_counts().head(10).plot(kind="bar")
    plt.title("Figure 6: Product Category Distribution")
    plt.tight_layout()
    plt.savefig("figures/fig6_product_category.png")
    plt.show()


# =========================================================
# FIGURE 7: CHANNEL DISTRIBUTION
# =========================================================
if "ChannelId" in df.columns:
    plt.figure(figsize=(6,4))
    df["ChannelId"].value_counts().plot(kind="bar")
    plt.title("Figure 7: Channel Distribution")
    plt.tight_layout()
    plt.savefig("figures/fig7_channel.png")
    plt.show()


# =========================================================
# FIGURE 8: TIME SERIES
# =========================================================
if "TransactionStartTime" in df.columns:
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")

    ts = df.groupby(df["TransactionStartTime"].dt.date)["Amount"].sum()

    plt.figure(figsize=(10,4))
    ts.plot()
    plt.title("Figure 8: Transaction Over Time")
    plt.tight_layout()
    plt.savefig("figures/fig8_timeseries.png")
    plt.show()


# =========================================================
# FIGURE 9: CORRELATION HEATMAP
# =========================================================
num_df = df.select_dtypes(include=["number"])

if num_df.shape[1] > 1:
    plt.figure(figsize=(8,5))
    sns.heatmap(num_df.corr(), cmap="coolwarm", annot=False)
    plt.title("Figure 9: Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("figures/fig9_correlation.png")
    plt.show()
# =============================
# SUMMARY STATISTICS (REQUIRED)
# =============================

num_df = df.select_dtypes(include=["number"])

summary = num_df.describe().T

summary["median"] = num_df.median()
summary["skew"] = num_df.skew()
summary["missing_rate"] = num_df.isnull().mean()

print("\n=== SUMMARY STATISTICS ===\n")
print(summary)

# Save for report
os.makedirs("figures", exist_ok=True)
summary.to_csv("figures/summary_statistics.csv")
print("\nTOP SKEWED FEATURES:\n")
print(summary.sort_values("skew", ascending=False).head(10))