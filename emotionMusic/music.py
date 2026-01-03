import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import os

st.set_page_config(page_title="Emotion Music Recommender")

st.title("Emotion Based Music Recommender")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "liveEmoji", "model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "liveEmoji", "labels.npy")
EMOTION_PATH = os.path.join(BASE_DIR, "emotion.npy")

# --- Load model safely ---
model = load_model(MODEL_PATH)
label = np.load(LABEL_PATH)

if "run" not in st.session_state:
    st.session_state["run"] = True

# --- Load emotion safely ---
emotion = ""
if os.path.exists(EMOTION_PATH):
    try:
        emotion = np.load(EMOTION_PATH)[0]
    except:
        emotion = ""

class EmotionProcessor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic()
        self.hands = mp.solutions.hands
        self.drawing = mp.solutions.drawing_utils

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = self.holistic.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0]*42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0]*42)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst, verbose=0))]

            cv2.putText(frm, pred, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            np.save(EMOTION_PATH, np.array([pred]))

        self.drawing.draw_landmarks(
            frm, res.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION)
        self.drawing.draw_landmarks(
            frm, res.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        self.drawing.draw_landmarks(
            frm, res.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Language")
singer = st.text_input("Singer")

if lang and singer and st.session_state["run"]:
    webrtc_streamer(
        key="emotion",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

if st.button("Recommend me songs"):
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = True
    else:
        webbrowser.open(
            f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}"
        )
        np.save(EMOTION_PATH, np.array([""]))
        st.session_state["run"] = False
