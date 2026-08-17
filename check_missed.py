import pandas as pd

known = pd.read_csv("demo_known_attack.csv")

# Known 1 = index 0, Known 5 = index 4
print("Known 1 attack type:", known.iloc[0]["label"])
print("Known 5 attack type:", known.iloc[4]["label"])

print("\nSaare known attack rows aur unke labels:")
print(known[["label"]])