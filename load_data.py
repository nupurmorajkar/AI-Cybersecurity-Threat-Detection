# src/load_data.py
# Loads the NSL-KDD dataset and assigns column names

import pandas as pd

def load_nslkdd(train_path="data/KDDTrain+.txt", test_path="data/KDDTest+.txt"):
    """
    NSL-KDD has no header row, so we define column names manually.
    The last two columns are: attack type and difficulty score.
    """

    # 41 features + attack label + difficulty score = 43 columns
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "attack_type", "difficulty"
    ]

    train_df = pd.read_csv(train_path, header=None, names=columns)
    test_df  = pd.read_csv(test_path,  header=None, names=columns)

    # Drop difficulty column — not needed for classification
    train_df.drop("difficulty", axis=1, inplace=True)
    test_df.drop("difficulty",  axis=1, inplace=True)

    # Convert attack types to binary: 'normal' = 0, anything else = 1
    train_df["label"] = train_df["attack_type"].apply(lambda x: 0 if x == "normal" else 1)
    test_df["label"]  = test_df["attack_type"].apply(lambda x: 0 if x == "normal" else 1)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"\nAttack distribution in training set:")
    print(train_df["label"].value_counts())

    return train_df, test_df