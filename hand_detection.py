import os
import numpy as np


# --------------------------------
# 1. Settings
# --------------------------------

dataset_path = "dataset"

labels = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "HELP",
    "PLEASE",
    "SORRY",
    "GOOD",
    "BAD",
    "I_LOVE_YOU"
]


# --------------------------------
# 2. Create empty lists
# --------------------------------

X = []
y = []


# --------------------------------
# 3. Load sequences
# --------------------------------

for label_index, label in enumerate(labels):

    label_path = os.path.join(
        dataset_path,
        label
    )

    print(f"\nLoading: {label}")

    files = [
        file
        for file in os.listdir(label_path)
        if file.endswith(".npy")
    ]

    # Use only 10 sequences from each sign
    files = files[:10]

    for file in files:

        file_path = os.path.join(
            label_path,
            file
        )

        sequence = np.load(file_path)

        # Check shape
        if sequence.shape == (30, 63):

            X.append(sequence)
            y.append(label_index)

            print(
                f"Loaded {file} → {sequence.shape}"
            )

        else:

            print(
                f"Skipped {file} → "
                f"wrong shape {sequence.shape}"
            )


# --------------------------------
# 4. Convert to NumPy arrays
# --------------------------------

X = np.array(X, dtype=np.float32)
y = np.array(y)


# --------------------------------
# 5. Display dataset information
# --------------------------------

print("\n==============================")
print("DATASET READY")
print("==============================")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nLabels:")

for index, label in enumerate(labels):
    print(index, "→", label)