import pandas as pd

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

df = pd.read_csv("data/KDDTrain+.txt", names=columns)   # "data/" add kiya
print(df.shape)
print(df["label"].unique())
hidden_attack_df = df[df["label"] == "neptune"]
known_attack_df = df[(df["label"] != "normal") & (df["label"] != "neptune")]
normal_test_df = df[df["label"] == "normal"].sample(20, random_state=42)

normal_test_df.to_csv("demo_normal.csv", index=False)
known_attack_df.sample(5, random_state=42).to_csv("demo_known_attack.csv", index=False)
hidden_attack_df.sample(3, random_state=42).to_csv("demo_hidden_attack.csv", index=False)

print("Demo files ban gayi!")
print("Normal rows:", len(normal_test_df))
print("Known attack rows available:", len(known_attack_df))
print("Hidden attack (neptune) rows available:", len(hidden_attack_df))