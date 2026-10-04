import sys
import pandas as pd
from scipy.stats import ks_2samp

ref = pd.read_csv("reference_data.csv")
new = ref.sample(150, replace=True, random_state=1).copy()
new["alcohol"] += 1.5  # simulated drift; remove this line for a no-drift demo

drifted = [c for c in ref.columns if ks_2samp(ref[c], new[c]).pvalue < 0.05]
print("Drifted features:", drifted)
open("drift_report.txt", "w").write("\n".join(drifted))
sys.exit(1 if len(drifted) > 3 else 0)
