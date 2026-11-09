import cv2
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------
# 1. Settings
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

sequence_length = 30
num_sequences = 10


# --------------------------------
# 2. Load MediaPipe hand model
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
# 3. Open webcam
# --------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    detector.close()
    exit()


# --------------------------------
# 4. Collect data for each sign
# --------------------------------

for label in labels:

    # Create folder
    save_path = os.path.join(
        "dataset",
        label
    )

    os.makedirs(
        save_path,
        exist_ok=True
    )


    # --------------------------------
    # Check existing sequences
    # --------------------------------

    existing_files = [
        file for file in os.listdir(save_path)
        if file.endswith(".npy")
    ]

    existing_count = len(existing_files)


    # --------------------------------
    # HELLO already has 50 samples
    # --------------------------------

    if label == "HELLO" and existing_count >= 20:

        print("\n--------------------------------")
        print("HELLO already has enough data.")
        print(f"Existing sequences: {existing_count}")
        print("Using 20 sequences later.")
        print("--------------------------------")

        continue


    # --------------------------------
    # Collect 20 sequences
    # --------------------------------

    sequence_number = existing_count

    while sequence_number < num_sequences:

        sequence = []


        print("\n--------------------------------")
        print(f"Sign: {label}")
        print(
            f"Sequence: "
            f"{sequence_number + 1}/{num_sequences}"
        )
        print("--------------------------------")

        input(
            "Press ENTER to start recording..."
        )


        # --------------------------------
        # Collect 30 frames
        # --------------------------------

        while len(sequence) < sequence_length:

            success, frame = cap.read()

            if not success:
                print("Could not access webcam")
                break


            # Mirror effect
            frame = cv2.flip(frame, 1)


            # Convert BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # --------------------------------
            # Convert to MediaPipe Image
            # --------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # --------------------------------
            # Detect hand
            # --------------------------------

            results = detector.detect(
                mp_image
            )


            # --------------------------------
            # Extract landmarks
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


                # Make sure there are 63 features
                if len(landmarks) == 63:

                    sequence.append(
                        landmarks
                    )


            # --------------------------------
            # Display information
            # --------------------------------

            cv2.putText(
                frame,
                f"Sign: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Frames: {len(sequence)}/{sequence_length}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.imshow(
                "Sign Language Data Collection",
                frame
            )


            # --------------------------------
            # Quit
            # --------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):

                cap.release()
                cv2.destroyAllWindows()
                detector.close()

                print("\nData collection stopped.")

                exit()


        # --------------------------------
        # Save sequence
        # --------------------------------

        if len(sequence) == sequence_length:

            sequence = np.array(
                sequence,
                dtype=np.float32
            )


            file_path = os.path.join(
                save_path,
                f"{sequence_number}.npy"
            )


            np.save(
                file_path,
                sequence
            )


            print(
                f"Saved: {file_path}"
            )

            print(
                "Shape:",
                sequence.shape
            )


            sequence_number += 1


# --------------------------------
# 5. Release resources
# --------------------------------

cap.release()
cv2.destroyAllWindows()

detector.close()


print("\n================================")
print("DATA COLLECTION COMPLETED!")
print("================================")