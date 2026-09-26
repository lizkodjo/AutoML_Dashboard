import numpy as np
import pandas as pd

# Dev utility: generate Excel test files in the current directory.

np.random.seed(42)
n = 100

df = pd.DataFrame(
    {
        "age": np.random.randint(20, 70, n),
        "salary": np.random.randint(30000, 150000, n),
        "department": np.random.choice(["IT", "HR", "Finance", "Marketing"], n),
        "experience": np.random.randint(0, 30, n),
        "target": np.random.randint(0, 2, n),
    }
)

df.to_excel("test_data.xlsx", index=False, sheet_name="Data")
print(f"Wrote test_data.xlsx ({len(df)} rows)")

with pd.ExcelWriter("test_data_multi.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Data")
    df.head(10).to_excel(writer, index=False, sheet_name="Preview")
    pd.DataFrame({"note": ["This file has 2 sheets"]}).to_excel(
        writer, index=False, sheet_name="Notes"
    )
print("Wrote test_data_multi.xlsx (3 sheets)")

df_missing = df.copy()
df_missing.loc[0:9, "salary"] = None
df_missing.loc[5:14, "department"] = None
df_missing.to_excel("test_data_missing.xlsx", index=False)
print("Wrote test_data_missing.xlsx (with NaN)")
