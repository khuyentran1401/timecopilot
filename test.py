from dotenv import load_dotenv

load_dotenv()

import nest_asyncio
import pandas as pd

from timecopilot import TimeCopilot
from timecopilot.models import AutoETS, SeasonalNaive
from timecopilot.models.foundation.chronos import Chronos

nest_asyncio.apply()

# Load PJM Hourly Energy Consumption dataset (last 3 months)
df = pd.read_csv(
    "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJM_Load_hourly.csv",
    parse_dates=["Datetime"],
)
df = df.sort_values("Datetime")
df = df[df["Datetime"] >= "2001-10-01"].reset_index(drop=True)

# Rename columns to TimeCopilot format
df = df.rename(columns={"Datetime": "ds", "PJM_Load_MW": "y"})
df["unique_id"] = "PJM"
df = df.sort_values("ds").reset_index(drop=True)

print(f"Shape: {df.shape}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

# Run TimeCopilot forecast
tc = TimeCopilot(
    llm="openai:gpt-4o",
    retries=3,
    forecasters=[AutoETS(), SeasonalNaive(), Chronos(repo_id="amazon/chronos-bolt-small")],
)
result = tc.forecast(df=df, freq="h")

# Inspect results
print("\n--- Time Series Features Analysis ---")
print(result.output.tsfeatures_analysis)

print("\n--- Cross-Validation Results ---")
print(result.output.cross_validation_results)

print("\n--- Model Comparison ---")
print(result.output.model_comparison)

print("\n--- Better than Seasonal Naive? ---")
print(result.output.is_better_than_seasonal_naive)

print("\n--- Selected Model ---")
print(result.output.selected_model)

print("\n--- Reason for Selection ---")
print(result.output.reason_for_selection)

print("\n--- Forecast DataFrame ---")
print(result.fcst_df.head())
