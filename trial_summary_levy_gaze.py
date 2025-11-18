import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the summary CSV (the one you pasted)
df = pd.read_csv("trial_summary_levy_gaze.csv")

# 2. Keep only trials with positive RT and path length
mask = (df["RT_ms"] > 0) & (df["path_len"] > 0)
sub = df.loc[mask].copy()

# 3. Take logs (natural log; you can use log10 if you prefer)
sub["log_RT"] = np.log(sub["RT_ms"])
sub["log_path"] = np.log(sub["path_len"])

# 4. Fit a straight line in log–log space
slope, intercept = np.polyfit(sub["log_path"], sub["log_RT"], 1)

# 5. Make the scatter + fitted line
plt.figure()
plt.scatter(sub["log_path"], sub["log_RT"], alpha=0.5)
x_line = np.linspace(sub["log_path"].min(), sub["log_path"].max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line)

plt.xlabel("log(path_len)")
plt.ylabel("log(RT_ms)")
plt.title(f"log–log RT vs path_len (slope ≈ {slope:.2f})")
plt.tight_layout()
plt.show()
