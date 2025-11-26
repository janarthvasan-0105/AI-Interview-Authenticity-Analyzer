import streamlit as st
import cv2
import face_alignment
import numpy as np
import time
import librosa
import speech_recognition as sr
from scipy.spatial import distance
from collections import deque
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import tempfile
from moviepy.editor import VideoFileClip
import shutil

# ==================================
# Video Analyzer
# ==================================
class VideoAnalyzer:
    def __init__(self):
        self.start_time = time.time()
        self.total_blinks = 0
        self.prev_ear = 0
        self.blink_thresh = 0.2275
        self.prev_head_dir = 0
        self.head_turn_thresh = 0.25
        self.sus_head_movements = 0
        self.gaze_history = deque(maxlen=10)
        self.gaze_warnings = 0
        self.center_x = 0.5
        self.calibrated = True

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def crop_eye(self, frame, eye_points):
        x_min = int(np.min(eye_points[:, 0]))
        x_max = int(np.max(eye_points[:, 0]))
        y_min = int(np.min(eye_points[:, 1]))
        y_max = int(np.max(eye_points[:, 1]))
        return frame[y_min:y_max, x_min:x_max]

    def detect_pupil(self, eye_img):
        if eye_img.size == 0: return None
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0: return None
        c = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        return (x + w // 2, y + h // 2)

    def get_head_direction(self, landmarks):
        nose, chin = landmarks[30], landmarks[8]
        return np.arctan2(chin[1] - nose[1], chin[0] - nose[0])

    def process_landmarks(self, frame, landmarks):
        warnings = []
        left_eye, right_eye = landmarks[36:42], landmarks[42:48]

        # Blink detection
        ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
        if ear < self.blink_thresh and self.prev_ear >= self.blink_thresh:
            self.total_blinks += 1
            warnings.append("BLINK")
        self.prev_ear = ear

        # Head movement
        head_dir = self.get_head_direction(landmarks)
        if self.prev_head_dir != 0 and abs(head_dir - self.prev_head_dir) > self.head_turn_thresh:
            self.sus_head_movements += 1
            warnings.append("HEAD MOVEMENT")
        self.prev_head_dir = head_dir

        # Eye gaze
        for eye_points in [left_eye, right_eye]:
            eye_img = self.crop_eye(frame, eye_points)
            pupil = self.detect_pupil(eye_img)
            if pupil is not None and eye_img.shape[1] > 0:
                cx, _ = pupil
                norm_x = cx / (eye_img.shape[1] + 1e-6)
                self.gaze_history.append(norm_x)
        
        if len(self.gaze_history) > 0:
            avg_x = np.mean(self.gaze_history)
            if avg_x < self.center_x - 0.1 or avg_x > self.center_x + 0.06:
                self.gaze_warnings += 1
                warnings.append("GAZE WARNING")
        
        return warnings

    def final_report(self, total_time):
        prob_ai = ((self.gaze_warnings * 0.5) + (self.sus_head_movements * 0.3) + (self.total_blinks * 0.2))
        prob_ai = min(prob_ai / 100.0, 1.0)
        return {
            "Total Time (s)": round(total_time, 2),
            "Total Blinks": self.total_blinks,
            "Suspected Head Movements": self.sus_head_movements,
            "Gaze Warnings": self.gaze_warnings,
            "AI-Use Probability": round(prob_ai, 2)
        }

# ==================================
# Speech Analyzer
# ==================================
class SpeechAnalyzer:
    def __init__(self):
        self.score = 0.0
    def analyze_audio(self, filename="output_audio.wav"):
        try:
            r = sr.Recognizer()
            with sr.AudioFile(filename) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            y, sr_rate = librosa.load(filename, sr=None)
            pitch = librosa.yin(y, fmin=50, fmax=300)
            pitch_std = np.std(pitch[~np.isnan(pitch)])
            if pitch_std < 5: self.score += 0.4
            if len(y) > 0 and (len(text.split()) / (len(y) / sr_rate)) < 1.5: self.score += 0.3
            if not text: self.score += 0.3
            return round(min(self.score, 1.0), 2), text
        except Exception as e:
            st.error(f"Speech analysis failed: {e}. Returning default values.")
            return 0.5, ""

# ==================================
# Text Analyzer
# ==================================
class TextAnalyzer:
    def __init__(self, model_path="./final_model"):
        self.detector = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.detector = pipeline("text-classification", model=model, tokenizer=tokenizer)
        except OSError:
            st.error(f"ERROR: Model not found at '{model_path}'. Text analysis will be skipped.")
    def analyze(self, text):
        if self.detector is None or not text.strip():
            return 0.5
        result = self.detector(text)[0]
        label, score = result['label'], result['score']
        st.write(f"**Text Analysis Prediction:** Label=`{label}`, Confidence=`{score:.2%}`")
        if "AI" in label.upper() or "MACHINE" in label.upper():
            return round(score, 2)
        else:
            return round(1 - score, 2)

# ==================================
# Fusion Layer
# ==================================
def fusion_layer(text_score, speech_score, video_score):
    overall_score = 0.4 * text_score + 0.4 * speech_score + 0.2 * video_score
    label = "Uncertain"
    if overall_score < 0.45: label = "Likely Human"
    elif overall_score > 0.65: label = "Likely AI / Suspicious"
    return {
        "text_score": text_score,
        "speech_score": speech_score,
        "video_score": video_score,
        "overall_score": round(overall_score, 2),
        "verdict": label
    }

# ==================================
# Main App Logic
# ==================================
st.set_page_config(layout="wide", page_title="AI Interview Analyzer")
st.title("📹📄🗣️ AI Interview Analyzer")
st.write("Upload an interview video to analyze for potential AI use based on video, speech, and text patterns.")

@st.cache_resource
def load_face_model():
    return face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

@st.cache_resource
def load_text_model():
    return TextAnalyzer(model_path="./final_model")

fa = load_face_model()
text_analyzer = load_text_model()

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    st.video(video_path)
    
    if st.button("Analyze Video"):
        video_analyzer = VideoAnalyzer()
        speech_analyzer = SpeechAnalyzer()
        
        with st.spinner('Analyzing video and creating annotated output... This may take a moment.'):
            temp_dir = tempfile.mkdtemp()
            annotated_video_path = None
            try:
                # --- Audio Extraction ---
                video_clip = VideoFileClip(video_path)
                audio_path = os.path.join(temp_dir, "output_audio.wav")
                video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
                total_duration = video_clip.duration

                # --- Video Processing with Annotation ---
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                annotated_video_path = os.path.join(temp_dir, "annotated_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (frame_width, frame_height))

                progress_bar = st.progress(0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                for i in range(frame_count):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    annotated_frame = frame.copy()
                    preds = fa.get_landmarks(annotated_frame)
                    
                    if preds is not None:
                        landmarks = preds[0]
                        warnings = video_analyzer.process_landmarks(frame, landmarks)
                        
                        for point in range(landmarks.shape[0]):
                            cv2.circle(annotated_frame, (int(landmarks[point,0]), int(landmarks[point,1])), 2, (0, 255, 0), -1)

                        if warnings:
                            y0, dy = 50, 40
                            for j, warning in enumerate(warnings):
                                y = y0 + j * dy
                                cv2.putText(annotated_frame, warning, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    out.write(annotated_frame)
                    progress_bar.progress((i + 1) / frame_count)
                
                cap.release()
                out.release()
                video_clip.close()
                st.success("Video analysis complete!")

                # --- Get and Display Reports ---
                st.subheader("📊 Analysis Reports")
                
                video_report = video_analyzer.final_report(total_duration)
                video_score = video_report["AI-Use Probability"]
                
                speech_score, transcript = speech_analyzer.analyze_audio(audio_path)
                if transcript:
                    st.write(f"**📝 Transcript:** *{transcript}*")
                else:
                    st.write("**📝 Transcript:** *Could not transcribe audio.*")

                text_score = text_analyzer.analyze(transcript)
                
                final_report = fusion_layer(text_score, speech_score, video_score)

                col1, col2 = st.columns(2)
                with col1:
                    st.json({"📹 Video Analysis Report": video_report}, expanded=True)
                with col2:
                    st.json({"🔥 Final Fusion Report": final_report}, expanded=True)

                # --- Display Annotated Video ---
                st.subheader("🎥 Annotated Video with Markings")
                
                # Use ffmpeg to convert to a web-friendly format
                final_video_path = os.path.join(temp_dir, "final_annotated_video.mp4")
                # Using -y to overwrite existing file, -c:v libx264 for compatibility
                os.system(f"ffmpeg -i {annotated_video_path} -y -c:v libx264 {final_video_path}")
                
                if os.path.exists(final_video_path):
                    st.video(final_video_path)
                else:
                    st.warning("Could not process annotated video for display. This might happen if ffmpeg is not installed.")


            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            finally:
                # Clean up temporary files
                if os.path.exists(video_path): os.remove(video_path)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

