import streamlit as st
import joblib
import re
import numpy as np
import cv2
import mediapipe as mp

from sentence_transformers import SentenceTransformer

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from streamlit_mic_recorder import speech_to_text


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Banking Query AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# THEME / CSS — dark "vault" theme, gold + teal accents
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #0A1120;
        --bg-panel: #131D30;
        --bg-panel-hover: #1B2A44;
        --border: #24344F;
        --text-primary: #EAF0F7;
        --text-secondary: #7C90AC;
        --accent-gold: #F2B84B;
        --accent-gold-dim: rgba(242, 184, 75, 0.15);
        --accent-teal: #35C7E0;
        --danger: #F0616B;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background: linear-gradient(160deg, #060A14 0%, #0D1830 40%, #142645 75%, #0A1120 100%);
        background-attachment: fixed;
    }

    section[data-testid="stSidebar"] {
        background-color: #060A14;
        border-right: 1px solid var(--border);
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* ---------------- Hero title ---------------- */

    .hero-wrap {
        text-align: center;
        padding: 8px 0 6px 0;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 46px;
        font-weight: 700;
        margin: 0;

        padding: 22px 32px;
        border-radius: 16px;

        background: linear-gradient(
            135deg,
            #0B1426 0%,
            #111E35 55%,
            #1C2940 100%
        );

        border: 1px solid rgba(245, 197, 66, 0.35);

        box-shadow:
            0 8px 30px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);

        color: #F5C542;

        transition:
            transform 0.3s ease,
            box-shadow 0.3s ease,
            border-color 0.3s ease,
            background 0.3s ease;
    }

    .hero-title:hover {
        transform: translateY(-5px);

        border-color: rgba(245, 197, 66, 0.7);

        background: linear-gradient(
            135deg,
            #0D172B 0%,
            #142442 55%,
            #202F49 100%
        );

        box-shadow:
            0 14px 40px rgba(0, 0, 0, 0.45),
            0 0 25px rgba(245, 197, 66, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }

    .hero-subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 16px;
        margin-top: 6px;
        margin-bottom: 28px;
    }

    /* ---------------- Metric cards ---------------- */

    div[data-testid="stMetric"] {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 14px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: var(--accent-gold);
        box-shadow: 0 12px 26px rgba(212, 169, 78, 0.18);
    }

    div[data-testid="stMetricValue"] {
        color: var(--accent-gold) !important;
    }

    /* ---------------- Section divider ---------------- */

    .section-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 34px 0 26px 0;
    }

    .section-label {
        font-size: 13px;
        letter-spacing: 0.04em;
        color: var(--text-secondary);
        margin-bottom: 4px;
    }

    /* ---------------- Generic panel ---------------- */

    .panel {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.30);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
        margin-bottom: 18px;
    }

    .panel:hover {
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.40);
    }

    /* ---------------- Prediction card ---------------- */

    .prediction-card {
        background: linear-gradient(160deg, var(--bg-panel) 0%, #12181F 100%);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-gold);
        border-radius: 16px;
        padding: 26px 28px;
        margin: 14px 0 22px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.40);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .prediction-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 36px rgba(212, 169, 78, 0.12);
    }

    .prediction-label {
        font-size: 13px;
        color: var(--text-secondary);
        letter-spacing: 0.03em;
        margin-bottom: 6px;
    }

    .prediction-intent {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 30px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 4px;
    }

    .confidence-text {
        color: var(--text-secondary);
        font-size: 15px;
    }

    .confidence-text b {
        color: var(--accent-teal);
    }

    /* ---------------- Custom animated bars ---------------- */

    .bar-track {
        width: 100%;
        height: 10px;
        background-color: #1C232B;
        border-radius: 6px;
        overflow: hidden;
        margin: 6px 0 14px 0;
        border: 1px solid var(--border);
    }

    .bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, var(--accent-gold), var(--accent-teal));
        width: 0%;
        animation: growBar 0.9s ease-out forwards;
    }

    @keyframes growBar {
        from { width: 0%; }
        to   { width: var(--w); }
    }

    .top-row {
        display: flex;
        justify-content: space-between;
        font-size: 14px;
        color: var(--text-primary);
        margin-top: 10px;
    }

    .top-row span.pct {
        color: var(--text-secondary);
    }

    /* ---------------- Buttons ---------------- */

    div.stButton > button {
        background-color: var(--bg-panel);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 8px 16px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        border-color: var(--accent-gold);
        box-shadow: 0 8px 20px rgba(212, 169, 78, 0.20);
        color: var(--accent-gold);
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, var(--accent-gold), #B98F3E);
        color: #14100A;
        border: none;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 10px 26px rgba(212, 169, 78, 0.35);
        color: #14100A;
    }

    /* ---------------- Inputs ---------------- */

    .stTextInput input, .stTextArea textarea {
        background-color: var(--bg-panel) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent-gold) !important;
        box-shadow: 0 0 0 1px var(--accent-gold) !important;
    }

    /* ---------------- Sidebar pipeline steps ---------------- */

    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 7px 10px;
        border-radius: 8px;
        margin-bottom: 3px;
        font-size: 13.5px;
        color: var(--text-secondary);
        transition: background-color 0.15s ease, color 0.15s ease;
    }

    .pipeline-step:hover {
        background-color: var(--bg-panel-hover);
        color: var(--text-primary);
    }

    .pipeline-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--accent-gold);
        flex-shrink: 0;
    }

    .sidebar-panel {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 16px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.30);
        transition: box-shadow 0.18s ease;
    }

    .sidebar-panel:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.40);
    }

    .sidebar-kv {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        padding: 3px 0;
        color: var(--text-secondary);
    }

    .sidebar-kv b {
        color: var(--text-primary);
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/best_model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
EMBEDDING_MODEL_PATH = "models/embedding_model"
HAND_MODEL_PATH = "models/hand_landmarker.task"


# ============================================================
# LOAD NLP MODELS
# ============================================================

@st.cache_resource
def load_nlp_models():

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    return model, label_encoder, embedding_model


@st.cache_resource
def load_hand_detector():

    base_options = python.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return vision.HandLandmarker.create_from_options(options)


try:
    model, label_encoder, embedding_model = load_nlp_models()

except Exception as e:
    st.error("❌ Unable to load the NLP model files.")
    st.code(str(e))
    st.info(
        "Make sure the following files/folders exist:\n"
        "- models/best_model.pkl\n"
        "- models/label_encoder.pkl\n"
        "- models/embedding_model/"
    )
    st.stop()


try:
    hand_detector = load_hand_detector()

except Exception as e:
    st.error("❌ Unable to load MediaPipe hand_landmarker.task.")
    st.code(str(e))
    st.info("Make sure this file exists:\nmodels/hand_landmarker.task")
    st.stop()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# NLP PREDICTION
# ============================================================

def predict_intent(text, top_n=3):

    text = str(text).strip()

    if not text:
        raise ValueError("The input text is empty.")

    processed_text = preprocess_text(text)

    if not processed_text:
        raise ValueError("The text became empty after preprocessing.")

    embedding = np.asarray(
        embedding_model.encode([processed_text], show_progress_bar=False)
    )

    prediction = model.predict(embedding)
    predicted_intent = label_encoder.inverse_transform(prediction)[0]

    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved model does not support probability prediction.")

    probabilities = model.predict_proba(embedding)[0]
    confidence = float(np.max(probabilities) * 100)

    top_indices = np.argsort(probabilities)[::-1][:top_n]
    top_predictions = []

    for index in top_indices:

        class_value = model.classes_[index] if hasattr(model, "classes_") else index
        intent = label_encoder.inverse_transform([class_value])[0]
        probability = float(probabilities[index] * 100)

        top_predictions.append({"intent": intent, "probability": probability})

    return predicted_intent, confidence, top_predictions, processed_text


# ============================================================
# MEDIAPIPE — GESTURE DETECTION
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def detect_hands(frame):

    if frame is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    return hand_detector.detect(mp_image)


def classify_gesture(landmarks):

    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    thumb_up = landmarks[4].y < landmarks[3].y

    if thumb_up and not (index_up or middle_up or ring_up or pinky_up):
        return "👍 Thumbs up"

    if index_up and middle_up and ring_up and pinky_up:
        return "✋ Open Palm"

    if not (index_up or middle_up or ring_up or pinky_up):
        return "✊ Fist"

    return "Unknown Gesture"


def detect_gesture(frame):

    result = detect_hands(frame)

    if not result or not result.hand_landmarks:
        return "No Hand Detected", result

    landmarks = result.hand_landmarks[0]

    return classify_gesture(landmarks), result


def draw_hand_landmarks(frame, result):

    if frame is None or result is None or not result.hand_landmarks:
        return frame

    height, width, _ = frame.shape

    for hand_landmarks in result.hand_landmarks:

        points = []

        for landmark in hand_landmarks:
            x = max(0, min(int(landmark.x * width), width - 1))
            y = max(0, min(int(landmark.y * height), height - 1))
            points.append((x, y))

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (79, 209, 197), 2)

        for x, y in points:
            cv2.circle(frame, (x, y), 5, (212, 169, 78), -1)

    return frame


# ============================================================
# UI HELPERS
# ============================================================

def render_bar(value):

    pct = max(0.0, min(value, 100.0))

    st.markdown(
        f"""
        <div class="bar-track">
            <div class="bar-fill" style="--w: {pct}%;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_prediction_result(predicted_intent, confidence, top_predictions, processed_text):

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">PREDICTED CATEGORY</div>
            <div class="prediction-intent">{predicted_intent}</div>
            <div class="confidence-text">Confidence: <b>{confidence:.2f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    render_bar(confidence)

    if confidence >= 80:
        st.success("🟢 High confidence prediction")
    elif confidence >= 60:
        st.warning("🟡 Moderate confidence prediction")
    else:
        st.info("🔵 Low confidence prediction — the query may be ambiguous.")

    with st.expander("🔧 View processed text"):
        st.write(processed_text)

    st.markdown("<div class='section-label'>TOP 3 PREDICTIONS</div>", unsafe_allow_html=True)

    for rank, item in enumerate(top_predictions, start=1):

        st.markdown(
            f"""
            <div class="top-row">
                <span>{rank}. {item['intent']}</span>
                <span class="pct">{item['probability']:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        render_bar(item["probability"])


def set_example(example):
    st.session_state["text_query"] = example


# ============================================================
# SESSION STATE DEFAULTS
# ============================================================

for key, default in [
    ("voice_text", ""),
    ("predicted_intent", ""),
    ("prediction_confidence", 0.0),
    ("gesture_confirmation", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">🏦 Banking Query AI</div>
        <div class="hero-subtitle">Speech-to-Text · Query Category Detection · Gesture Confirmation</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL METRICS
# ============================================================

st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Banking Intents", "77")

with col2:
    st.metric("Test Accuracy", "92.79%")

with col3:
    st.metric("Weighted F1", "92.77%")

with col4:
    st.metric("Model", "Logistic Regression")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ============================================================
# INPUT METHOD SELECTION
# ============================================================

st.markdown("### 🎯 Choose Input Method")

input_method = st.radio(
    "Select how you want to give your command:",
    ["💬 Type Command", "🎤 Voice Command"],
    horizontal=True,
    label_visibility="collapsed"
)

input_text = ""

if input_method == "💬 Type Command":

    st.markdown("#### 💬 Enter Banking Query")

    input_text = st.text_area(
        "Customer Query",
        placeholder="Example: Why hasn't my card arrived yet?",
        height=110,
        key="text_query",
        label_visibility="collapsed"
    )

    st.markdown("**💡 Example queries**")

    examples = [
        "Why hasn't my card arrived yet?",
        "I lost my card",
        "How can I change my PIN?",
        "Why was my cash withdrawal declined?",
        "I want to cancel a transfer",
        "How do I transfer money?",
        "I need a refund",
        "Why is my balance incorrect?",
    ]

    example_columns = st.columns(2)

    for i, example in enumerate(examples):
        with example_columns[i % 2]:
            st.button(
                example,
                key=f"example_{i}",
                use_container_width=True,
                on_click=set_example,
                args=(example,)
            )

    input_text = st.session_state.get("text_query", "")

else:

    st.markdown("#### 🎤 Speak Your Banking Query")

    st.info("💡 Allow microphone access in your browser when prompted.")

    voice_result = speech_to_text(
        language="en",
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True,
        key="voice_input"
    )

    if voice_result:
        st.session_state.voice_text = str(voice_result)

    if st.session_state.voice_text:
        st.markdown("**📝 Recognized Text**")
        st.info(st.session_state.voice_text)

    input_text = st.session_state.get("voice_text", "")


if st.button("🔍 Classify Query", type="primary", use_container_width=True):

    if not input_text.strip():
        st.warning("Please provide a banking query first.")

    elif len(input_text.strip()) < 3:
        st.warning("Please enter a longer query.")

    else:

        with st.spinner("Analyzing your query..."):

            try:
                predicted_intent, confidence, top_predictions, processed_text = predict_intent(input_text)

                st.session_state.predicted_intent = predicted_intent
                st.session_state.prediction_confidence = confidence

                display_prediction_result(
                    predicted_intent, confidence, top_predictions, processed_text
                )

            except Exception as e:
                st.error("❌ Unable to make a prediction.")
                st.code(str(e))


# ============================================================
# GESTURE CONFIRMATION
# ============================================================

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### 📸 Gesture Confirmation")
st.write("Show a hand gesture to the camera — the result is applied automatically from the photo.")

if not st.session_state.predicted_intent:

    st.info("🔍 Classify a query first to enable gesture confirmation.")

else:

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">CATEGORY TO CONFIRM</div>
            <div class="prediction-intent">{st.session_state.predicted_intent}</div>
            <div class="confidence-text">Confidence: <b>{st.session_state.prediction_confidence:.2f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        **👍 Thumbs up** → Confirms automatically &nbsp;&nbsp;·&nbsp;&nbsp;
        **✋ Open palm** → Rejects automatically &nbsp;&nbsp;·&nbsp;&nbsp;
        **✊ Fist** → No action
        """
    )

    camera_image = st.camera_input("📸 Take a picture of your hand")

    if camera_image is not None:

        image_bytes = camera_image.getvalue()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Unable to process the camera image. Please try again.")

        else:

            gesture, detection_result = detect_gesture(frame)
            output_frame = draw_hand_landmarks(frame.copy(), detection_result)
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            st.image(output_frame_rgb, caption="Hand Landmark Detection", use_container_width=True)

            st.markdown("#### ✋ Detected Gesture")

            if gesture == "👍 Thumbs up":
                st.session_state.gesture_confirmation = "Confirmed"
                st.success(f"👍 Thumbs up detected — category confirmed: {st.session_state.predicted_intent}")

            elif gesture == "✋ Open Palm":
                st.session_state.gesture_confirmation = "Rejected"
                st.error(f"✋ Open palm detected — category rejected: {st.session_state.predicted_intent}")

            elif gesture == "✊ Fist":
                st.session_state.gesture_confirmation = "No Action"
                st.info("✊ Fist detected — no action taken.")

            else:
                st.warning("⚠️ No clear gesture detected. Please try again.")


if st.session_state.gesture_confirmation:

    st.markdown("### 📋 Confirmation Status")

    if st.session_state.gesture_confirmation == "Confirmed":
        st.success("✅ Category confirmed successfully.")
    elif st.session_state.gesture_confirmation == "Rejected":
        st.error("❌ Category rejected.")
    elif st.session_state.gesture_confirmation == "No Action":
        st.info("✊ No action was taken.")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏦 Banking Query AI")

    st.write(
        "This application predicts the category of customer "
        "banking queries using the BANKING77 dataset."
    )

    st.markdown("<hr class='section-divider' style='margin:18px 0;'>", unsafe_allow_html=True)

    st.markdown("#### 🤖 NLP Model")

    st.markdown(
        """
        <div class="sidebar-panel">
            <div class="sidebar-kv"><span>Final Model</span><b>Tuned Logistic Regression</b></div>
            <div class="sidebar-kv"><span>Embedding</span><b>all-MiniLM-L6-v2</b></div>
            <div class="sidebar-kv"><span>Test Accuracy</span><b>92.79%</b></div>
            <div class="sidebar-kv"><span>Weighted F1</span><b>92.77%</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### 🎤 Speech Pipeline")

    speech_steps = ["Speech", "Text", "Preprocessing", "Sentence Embedding", "Logistic Regression", "Category"]

    steps_html = "".join(
        f"<div class='pipeline-step'><span class='pipeline-dot'></span>{step}</div>"
        for step in speech_steps
    )

    st.markdown(f"<div class='sidebar-panel'>{steps_html}</div>", unsafe_allow_html=True)

    st.markdown("#### 📷 MediaPipe")

    mp_steps = ["Webcam", "Camera Image", "OpenCV", "MediaPipe", "21 Hand Landmarks", "Hand Detection"]

    mp_steps_html = "".join(
        f"<div class='pipeline-step'><span class='pipeline-dot'></span>{step}</div>"
        for step in mp_steps
    )

    st.markdown(f"<div class='sidebar-panel'>{mp_steps_html}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider' style='margin:18px 0;'>", unsafe_allow_html=True)

    st.caption("BANKING77 NLP Project")