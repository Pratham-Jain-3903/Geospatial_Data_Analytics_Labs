import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os  
from sklearn.linear_model import LinearRegression

# Define the dataset path
dataset_path = r"C:\Users\Pratham Jain\SisterDear\Geospatial\DemoCSV\Complete_data_new.csv"

# Check if the file exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The file at {dataset_path} does not exist.")

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(dataset_path)

# Ensure we only take the first 1433 rows if available
df = df.head(1433)

# Print the first few rows of the DataFrame
print(df.head())

# Create a Time column and Lag_1 column
df["Time"] = np.arange(len(df.index))
df['Lag_1'] = df['Solar energy Generation  (kWh)'].shift(1)

# Reorder columns
df = df[['Solar energy Generation  (kWh)', 'Lag_1', 'Time']]

# Check for missing values in 'Solar energy Generation  (kWh)' and drop if any
if df['Solar energy Generation  (kWh)'].isnull().any():
    print("Warning: Missing values found in 'Solar energy Generation  (kWh)'. They will be dropped.")
    df = df.dropna(subset=['Solar energy Generation  (kWh)'])

# Ensure Lag_1 is also valid
if df['Lag_1'].isnull().any():
    df = df.dropna(subset=['Lag_1'])

# Plot settings
plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight="bold",
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)

# Create the first plot for Time vs Solar Energy Generation
fig, ax = plt.subplots()
ax.plot(df['Time'], df['Solar energy Generation  (kWh)'], color="0.75", label='Solar Energy Generation (kWh)')
sns.regplot(
    x="Time", y="Solar energy Generation  (kWh)", data=df, ci=None, scatter_kws=dict(color="0.25"), ax=ax
)

# Set titles and labels
ax.set_title("Time Plot of Solar Energy Generation")
ax.set_xlabel("Time")
ax.set_ylabel("Solar Energy Generation (kWh)")
ax.legend()

# Create the second plot for Lag_1 vs Solar Energy Generation
fig, ax = plt.subplots()
ax.plot(df['Lag_1'], df['Solar energy Generation  (kWh)'], color="0.75", label='Solar Energy Generation (kWh)')
sns.regplot(
    x="Lag_1", y="Solar energy Generation  (kWh)", data=df, ci=None, scatter_kws=dict(color="0.25"), ax=ax
)

# Set titles and labels
ax.set_title("Lag_1 vs Solar Energy Generation")
ax.set_xlabel("Lag_1 (kWh)")
ax.set_ylabel("Solar Energy Generation (kWh)")
ax.legend()

# Prepare data for model training
X = df[['Time']].dropna()  # features
y = df['Solar energy Generation  (kWh)']  # target
y, X = y.align(X, join='inner') 

# Train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
df['pred'] = np.nan  # Initialize the predictions column
df.loc[X.index, 'pred'] = y_pred  # Align predictions with the original DataFrame

# Create a plot for predictions
fig, ax = plt.subplots()
ax.plot(df['Time'], df['pred'], color="0.75", label='Predicted Solar Energy Generation (kWh)', linestyle='--')
ax.plot(df['Time'], df['Solar energy Generation  (kWh)'], color="0.25", label='Actual Solar Energy Generation (kWh)')
ax.set_title("Predicted vs Actual Solar Energy Generation")
ax.set_xlabel("Time")
ax.set_ylabel("Solar Energy Generation (kWh)")
ax.legend()

# Show all plots
plt.tight_layout()
plt.show()
