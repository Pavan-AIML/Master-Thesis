import pandas as pd
from pathlib import Path

path = Path("20260205-030517") / "final_results_summery.csv"
results = pd.read_csv(path)
results
