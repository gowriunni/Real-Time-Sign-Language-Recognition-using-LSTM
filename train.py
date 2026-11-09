import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --------------------------------
# 1. Load dataset
# --------------------------------

X = np.load("X.npy")
y = np.load("y.npy")


print("Original dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# --------------------------------
# 2. Split dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n==============================")
print("TRAIN / TEST SPLIT")
print("==============================")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# --------------------------------
# 3. Build LSTM model
# --------------------------------

model = Sequential([

    LSTM(
        128,
        return_sequences=True,
        input_shape=(30, 63)
    ),

    Dropout(0.3),

    LSTM(64),

    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(10, activation="softmax")
])


# --------------------------------
# 4. Display model
# --------------------------------

print("\n==============================")
print("LSTM MODEL")
print("==============================")

model.summary()


# --------------------------------
# 5. Compile model
# --------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------
# 6. Train model
# --------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=8,
    validation_split=0.2
)


# --------------------------------
# 7. Evaluate model
# --------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)
# --------------------------------
# 7. Detailed evaluation
# --------------------------------

y_pred_prob = model.predict(X_test)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)


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


print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=labels,
        zero_division=0
    )
)


# --------------------------------
# 8. Confusion matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title("Sign Language Confusion Matrix")

plt.colorbar()

plt.xticks(
    range(len(labels)),
    labels,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(labels)),
    labels
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.show()


print("\n==============================")
print("TEST RESULTS")
print("==============================")

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# --------------------------------
# 8. Save model
# --------------------------------

model.save(
    "models/sign_lstm.keras"
)

print("\nModel saved successfully!")