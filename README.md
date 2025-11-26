# 🎥🗣️📄 AI Interview Authenticity Analyzer

> A multimodal system that flags whether an interview response looks **human** or **AI-assisted** – using **video**, **audio**, and **text** together.

We built this system in **24 hours** during a hack-style sprint to tackle a growing problem:  
it's getting harder to tell if someone is genuinely answering in an interview or secretly using AI to generate or feed them responses.

To tackle this, we designed a **complete multimodal authenticity detection pipeline**:
- A **video model** to track **eye gaze**, **head movement**, and **blinks**
- An **audio model** to analyze **speech patterns** and extract a **transcript**
- A **text model** to check if the **transcript matches AI-generated patterns**
- Finally, a **fusion layer** that combines all three into one **authenticity score**

---
[FOLDER STRUCTURE]
ai-interview-authenticity-analyzer/
├── app.py                     # Main Streamlit app (the code you pasted)
├── final_model/               # Trained text classification model
│   ├── config.json
│   ├── tokenizer.json
│   ├── vocab.txt / merges.txt (depending on model)
│   ├── special_tokens_map.json
│   ├── pytorch_model.bin
│   └── ...                    # Any other HF model files
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── assets/
│   ├── sample_video.mp4       # Optional sample video
│   └── screenshots/           # Optional UI screenshots
└── .streamlit/                # (Optional) Streamlit config
    └── config.toml

## 🚀 What This App Does

Given an **interview video**, the app:

1. **Extracts the audio** from the video.
2. **Transcribes speech** to text.
3. **Analyzes the video feed** for:
   - Blink frequency
   - Suspicious head movements
   - Eye-gaze deviation
4. **Analyzes the audio** for:
   - Pitch variation (monotone / flat vs natural)
   - Speaking rate vs duration
5. **Analyzes the text** for:
   - Whether the response looks like it was written by an AI model
6. **Fuses all signals** into one **final verdict**:
   - `Likely Human`
   - `Uncertain`
   - `Likely AI / Suspicious`
7. Generates an **annotated video** with:
   - Facial landmarks drawn
   - On-screen warnings like `BLINK`, `HEAD MOVEMENT`, `GAZE WARNING`

All of this is wrapped in a **Streamlit UI** so you can just upload a video and click a button.

---

## 🧱 System Architecture (High-Level)

**Inputs:**  
🎞️ Interview video (`.mp4`, `.mov`, `.avi`)

**Parallel Pipelines:**

1. **Video Pipeline (Non-verbal behavior)**
   - Face landmarks via `face_alignment`
   - Blink detection using **Eye Aspect Ratio (EAR)**
   - Head direction changes via nose–chin angle
   - Eye-gaze drift using pupil position within the eye crop

2. **Audio Pipeline (Voice behavior)**
   - Extracts audio using `moviepy`
   - Transcribes using `speech_recognition` (Google API)
   - Uses `librosa` to:
     - Estimate pitch
     - Measure pitch variation (natural vs robotic)
     - Compare speaking rate vs audio length

3. **Text Pipeline (Linguistic behavior)**
   - Runs transcript through a **fine-tuned text classifier** (`./final_model`)
   - Returns a **probability that the text is AI-generated**

4. **Fusion Layer**
   - Combines:
     - `text_score` (0–1)
     - `speech_score` (0–1)
     - `video_score` (0–1)
   - Weighted fusion:
     ```python
     overall_score = 0.4 * text_score + 0.4 * speech_score + 0.2 * video_score
     ```
   - Verdict logic:
     - `< 0.45` → **Likely Human**
     - `0.45–0.65` → **Uncertain**
     - `> 0.65` → **Likely AI / Suspicious**

---

## 🧠 Core Logic & Components

2️⃣ SpeechAnalyzer – Voice & Rhythm

>>Defined in code as:  class SpeechAnalyzer:
Steps:

1.Uses speech_recognition:

2.Loads the extracted .wav audio file.

3.Calls Google’s speech API to get the transcript.

4.Loads audio using librosa:

5.Computes pitch using librosa.yin.

6.Measures pitch standard deviation:

-----Low variance → voice is too flat → add to AI suspicion.

>>Compares:

Number of words in transcript

Audio duration - If words-per-second is too low → adds to suspicion.

Builds a speech_score between 0 and 1: Starts with 0.0

Adds: +0.4 if pitch variation is very low ,+0.3 if speech density is abnormal ,+0.3 if no text could be recognized

Returns: speech_score (0–1) ,transcript (string)

3️⃣ TextAnalyzer – AI Text Detection

>>Defined in code as:
class TextAnalyzer:
    def __init__(self, model_path="./final_model"):
        ...
>>What it does:

1.Loads a Hugging Face transformer model from ./final_model:

2.AutoTokenizer

3.AutoModelForSequenceClassification

4.Wraps them in a pipeline("text-classification").

>>Inference:

Given a transcript text, it returns: label (e.g., "AI-GENERATED" or "HUMAN", depending on your model), score (confidence between 0–1)

If the label suggests AI: text_score = score

If the label suggests human: text_score = 1 - score

If no model or empty text: returns 0.5 (neutral).

4️⃣ fusion_layer – Final Decision Engine

>>Defined in code as:
def fusion_layer(text_score, speech_score, video_score):
    overall_score = 0.4 * text_score + 0.4 * speech_score + 0.2 * video_score
    ...

>>Outputs: text_score ,speech_score ,video_score ,overall_score

>>verdict ∈ { "Likely Human", "Uncertain", "Likely AI / Suspicious" }

This fusion is what makes the system multimodal instead of relying on a single weak signal.
