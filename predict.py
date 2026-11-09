import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------
# 1. Load trained LSTM model
# --------------------------------

model = tf.keras.models.load_model(
    "models/sign_lstm.keras"
)


# --------------------------------
# 2. Sign labels
# --------------------------------

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
# 3. Load MediaPipe hand model
# --------------------------------

base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# --------------------------------
# 4. Open webcam
# --------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open webcam")

    detector.close()

    exit()


# --------------------------------
# 5. Sequence
# --------------------------------

sequence = []

sequence_length = 30


# --------------------------------
# 6. Real-time prediction
# --------------------------------

while True:

    success, frame = cap.read()

    if not success:

        print("Could not read webcam")

        break


    # Mirror effect
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------
    # 7. Convert to MediaPipe image
    # --------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------
    # 8. Detect hand
    # --------------------------------

    results = detector.detect(
        mp_image
    )


    # --------------------------------
    # 9. Extract landmarks
    # --------------------------------

    if results.hand_landmarks:

        hand = results.hand_landmarks[0]

        landmarks = []

        for landmark in hand:

            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])


        if len(landmarks) == 63:

            sequence.append(landmarks)


    # --------------------------------
    # 10. Keep only last 30 frames
    # --------------------------------

    if len(sequence) > sequence_length:

        sequence = sequence[-sequence_length:]


    # --------------------------------
    # 11. Make prediction
    # --------------------------------

    if len(sequence) == sequence_length:

        input_data = np.array(
            sequence,
            dtype=np.float32
        )

        # Add batch dimension
        input_data = np.expand_dims(
            input_data,
            axis=0
        )

        # Shape becomes:
        # (1, 30, 63)

        predictions = model.predict(
            input_data,
            verbose=0
        )


        # Get highest probability
        predicted_index = np.argmax(
            predictions[0]
        )

        confidence = predictions[0][
            predicted_index
        ]


        predicted_label = labels[
            predicted_index
        ]


        # --------------------------------
        # 12. Display prediction
        # --------------------------------

        cv2.putText(
            frame,
            f"Sign: {predicted_label}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # --------------------------------
    # 13. Display webcam
    # --------------------------------

    cv2.imshow(
        "Sign Language Recognition",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# --------------------------------
# 14. Release resources
# --------------------------------

cap.release()

cv2.destroyAllWindows()

detector.close()