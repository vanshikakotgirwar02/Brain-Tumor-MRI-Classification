import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "custom_cnn.keras"

COMPARISON_CHART = (
    BASE_DIR
    / "results"
    / "plots"
    / "model_accuracy_comparison.png"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(37, 99, 235, 0.13),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(6, 182, 212, 0.10),
            transparent 28%
        ),
        #080d1c;
    color: #f8fafc;
}

header {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background: #0d1428;
    border-right: 1px solid rgba(148, 163, 184, 0.14);
}

section[data-testid="stSidebar"] * {
    box-sizing: border-box;
}

.sidebar-logo {
    text-align: center;
    padding: 12px 0 22px 0;
}

.sidebar-brain {
    font-size: 52px;
    line-height: 1;
    margin-bottom: 12px;
}

.sidebar-title {
    color: #ffffff !important;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: 0.2px;
}

.sidebar-subtitle {
    color: #8fa4c7 !important;
    font-size: 13px;
    margin-top: 5px;
}


/* Sidebar section label */

section[data-testid="stSidebar"] .stCaption {
    color: #64748b !important;
}


/* Radio navigation */

section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 7px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 12px;
    padding: 10px 12px;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(59, 130, 246, 0.12);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #cbd5e1 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(
    input:checked
) {
    background: linear-gradient(
        90deg,
        rgba(37, 99, 235, 0.28),
        rgba(6, 182, 212, 0.12)
    );
    border: 1px solid rgba(96, 165, 250, 0.25);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(
    input:checked
) p {
    color: #ffffff !important;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero-title {
    font-size: 55px;
    font-weight: 850;
    line-height: 1.05;
    margin-top: 10px;
    margin-bottom: 12px;

    background: linear-gradient(
        90deg,
        #ffffff 0%,
        #60a5fa 55%,
        #22d3ee 100%
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #a9b8d1;
    font-size: 19px;
    line-height: 1.65;
    max-width: 850px;
}


/* ==========================================================
   SECTION HEADINGS
   ========================================================== */

.section-title {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 800;
    margin-top: 28px;
    margin-bottom: 8px;
}

.section-description {
    color: #94a3b8;
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 22px;
}


/* ==========================================================
   CARDS
   ========================================================== */

.card {
    background: rgba(17, 27, 52, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 25px;
    margin: 8px 0;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.18);
}

.card-title {
    color: #f1f5f9;
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 9px;
}

.card-text {
    color: #aebbd0;
    font-size: 15px;
    line-height: 1.7;
}


/* ==========================================================
   METRIC CARDS
   ========================================================== */

.metric-card {
    background: linear-gradient(
        145deg,
        rgba(28, 43, 80, 0.96),
        rgba(13, 22, 43, 0.96)
    );

    border: 1px solid rgba(96, 165, 250, 0.20);
    border-radius: 18px;
    padding: 23px 15px;
    text-align: center;
    min-height: 125px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.16);
}

.metric-number {
    color: #60a5fa;
    font-size: 31px;
    font-weight: 850;
}

.metric-label {
    color: #aebbd0;
    font-size: 14px;
    margin-top: 6px;
}


/* ==========================================================
   UPLOAD AREA
   ========================================================== */

[data-testid="stFileUploader"] {
    margin-top: 15px;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(17, 27, 52, 0.88) !important;
    border: 1px dashed rgba(96, 165, 250, 0.55) !important;
    border-radius: 18px !important;
    padding: 22px !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #1d4ed8 !important;
    color: #ffffff !important;
}

[data-testid="stFileUploaderDropzone"] small {
    color: #94a3b8 !important;
}


/* ==========================================================
   PREDICTION CARD
   ========================================================== */

.prediction-box {
    background:
        linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.23),
            rgba(6, 182, 212, 0.13)
        );

    border: 1px solid rgba(96, 165, 250, 0.38);
    border-radius: 22px;
    padding: 30px 20px;
    text-align: center;
    margin-top: 20px;
}

.prediction-label {
    color: #9fb1d1;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.prediction-result {
    color: #ffffff;
    font-size: 36px;
    font-weight: 850;
    margin: 9px 0;
}

.confidence {
    color: #67e8f9;
    font-size: 19px;
    font-weight: 700;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.custom-footer {
    text-align: center;
    color: #64748b;
    padding: 45px 0 15px 0;
    font-size: 13px;
}


/* ==========================================================
   ALERTS
   ========================================================== */

div[data-testid="stAlert"] {
    border-radius: 14px;
}


/* ==========================================================
   PROGRESS BARS
   ========================================================== */

div[data-testid="stProgressBar"] {
    margin-bottom: 12px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    model_available = True
    model_error = None

except Exception as e:
    model_available = False
    model_error = str(e)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_class_name(class_name):
    return class_name.replace("_", " ").title()


def predict_image(image):

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[predicted_index]

    return predicted_class, confidence, predictions


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-logo">
    <div class="sidebar-brain">🧠</div>
    <div class="sidebar-title">NeuroVision AI</div>
    <div class="sidebar-subtitle">Brain MRI Classification</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div style="color:#64748b;font-size:11px;font-weight:800;letter-spacing:1.5px;margin-bottom:8px;">NAVIGATION</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔬 MRI Classifier",
            "📊 Model Insights"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        """
<div style="padding:8px 2px;">
    <div style="color:#e2e8f0;font-weight:700;font-size:13px;">
        AI-assisted research tool
    </div>
    <div style="color:#7f91b0;font-size:12px;margin-top:8px;line-height:1.6;">
        Built with Python, TensorFlow and Streamlit.
    </div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 1 — HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="hero-title">NeuroVision AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">An AI-powered brain MRI image classification system designed to explore deep learning for medical image analysis.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
<div class="metric-card">
    <div class="metric-number">4</div>
    <div class="metric-label">MRI Classes</div>
</div>
""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
<div class="metric-card">
    <div class="metric-number">2,443</div>
    <div class="metric-label">MRI Images</div>
</div>
""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
<div class="metric-card">
    <div class="metric-number">81.71%</div>
    <div class="metric-label">Test Accuracy</div>
</div>
""",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
<div class="metric-card">
    <div class="metric-number">13.2M</div>
    <div class="metric-label">CNN Parameters</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">What does NeuroVision AI do?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Upload a brain MRI image and the trained deep learning model predicts one of four categories.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
<div class="card">
    <div class="card-title">🔬 Image Classification</div>
    <div class="card-text">
        The system analyzes an uploaded MRI image and predicts
        whether it belongs to glioma, meningioma, pituitary,
        or no tumor.
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            """
<div class="card">
    <div class="card-title">⚙️ Deep Learning Pipeline</div>
    <div class="card-text">
        The project includes image preprocessing, augmentation,
        a custom CNN, transfer learning and model evaluation.
    </div>
</div>
""",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
<div class="card">
    <div class="card-title">📈 Model Evaluation</div>
    <div class="card-text">
        Two deep learning approaches were evaluated using
        accuracy, loss, precision, recall, F1-score and
        confusion matrices.
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            """
<div class="card">
    <div class="card-title">🚀 Interactive Application</div>
    <div class="card-text">
        A Streamlit interface makes the trained model
        accessible through a simple image-upload workflow.
    </div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="custom-footer">NeuroVision AI • Deep Learning for Brain MRI Image Classification</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2 — MRI CLASSIFIER
# ============================================================

elif page == "🔬 MRI Classifier":

    st.markdown(
        '<div class="section-title">🔬 MRI Image Classifier</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Upload a brain MRI image and let the trained Custom CNN generate a classification prediction.</div>',
        unsafe_allow_html=True
    )

    if not model_available:

        st.error("The trained model could not be loaded.")

        st.code(model_error)

        st.stop()

    uploaded_file = st.file_uploader(
        "Choose a brain MRI image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear brain MRI image."
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        predicted_class, confidence, probabilities = predict_image(
            image
        )

        formatted_class = format_class_name(
            predicted_class
        )

        col1, col2 = st.columns([1.05, 0.95])

        with col1:

            st.markdown(
                '<div class="card-title">🖼️ Uploaded MRI</div>',
                unsafe_allow_html=True
            )

            st.image(
                image,
                caption="Uploaded MRI",
                use_container_width=True
            )

        with col2:

            st.markdown(
                f"""
<div class="prediction-box">
    <div class="prediction-label">MODEL PREDICTION</div>
    <div class="prediction-result">{formatted_class}</div>
    <div class="confidence">
        Confidence: {confidence * 100:.2f}%
    </div>
</div>
""",
                unsafe_allow_html=True
            )

            st.write("")

            if confidence >= 0.80:
                st.success("High model confidence")

            elif confidence >= 0.60:
                st.warning("Moderate model confidence")

            else:
                st.warning("Low model confidence")

        st.markdown(
            '<div class="section-title">📊 Prediction Probabilities</div>',
            unsafe_allow_html=True
        )

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):

            formatted_name = format_class_name(
                class_name
            )

            percentage = probability * 100

            st.write(
                f"**{formatted_name}** — {percentage:.2f}%"
            )

            st.progress(
                float(probability)
            )

        st.warning(
            "⚠️ Educational/research use only. This model is not "
            "a medical diagnostic system and should not replace "
            "evaluation by a qualified medical professional."
        )

    else:

        st.markdown(
            """
<div class="card" style="text-align:center;padding:55px 25px;">
    <div style="font-size:52px;">🧠</div>
    <div class="card-title">Ready to analyze an MRI?</div>
    <div class="card-text">
        Choose an MRI image using the upload box above
        to begin AI classification.
    </div>
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# PAGE 3 — MODEL INSIGHTS
# ============================================================

elif page == "📊 Model Insights":

    st.markdown(
        '<div class="section-title">📊 Model Insights Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Performance comparison between the Custom CNN and MobileNetV2 transfer learning model.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
<div class="card">
    <div class="card-title">🧠 Custom CNN</div>
    <div class="card-text">
        <b>Test Accuracy:</b> 81.71%<br>
        <b>Test Loss:</b> 0.4954<br>
        <b>Macro F1:</b> 81.05%<br>
        <b>Weighted F1:</b> 81.48%<br>
        <b>Parameters:</b> 13.23M
    </div>
</div>
""",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
<div class="card">
    <div class="card-title">⚡ MobileNetV2</div>
    <div class="card-text">
        <b>Test Accuracy:</b> 80.89%<br>
        <b>Test Loss:</b> 0.5010<br>
        <b>Macro F1:</b> 80.45%<br>
        <b>Weighted F1:</b> 80.56%<br>
        <b>Parameters:</b> 2.42M
    </div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">🏆 Model Comparison</div>',
        unsafe_allow_html=True
    )

    if COMPARISON_CHART.exists():

        st.image(
            str(COMPARISON_CHART),
            caption="Test Accuracy Comparison",
            use_container_width=True
        )

    else:

        st.info(
            "Model comparison chart was not found."
        )

    st.success(
        "🏆 Best model based on test accuracy: Custom CNN — 81.71%"
    )

    st.markdown(
        '<div class="section-title">📌 Class-wise F1 Scores</div>',
        unsafe_allow_html=True
    )

    class_data = {
        "Glioma": (88.89, 86.96),
        "Meningioma": (67.83, 67.89),
        "No Tumor": (77.06, 86.36),
        "Pituitary": (90.43, 80.60)
    }

    for class_name, scores in class_data.items():

        cnn_score = scores[0]
        mobile_score = scores[1]

        st.markdown(
            f"**{class_name}**"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.caption(
                f"Custom CNN — {cnn_score:.2f}%"
            )

            st.progress(
                cnn_score / 100
            )

        with col2:

            st.caption(
                f"MobileNetV2 — {mobile_score:.2f}%"
            )

            st.progress(
                mobile_score / 100
            )

    st.markdown(
        '<div class="section-title">💡 Key Observation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="card">
    <div class="card-text">
        The <b>Custom CNN</b> achieved slightly better overall
        test performance with 81.71% accuracy.
        <br><br>
        <b>MobileNetV2</b> used significantly fewer parameters,
        making it a lighter model.
        <br><br>
        The Custom CNN performed particularly well on
        <b>glioma</b> and <b>pituitary</b>, while MobileNetV2
        performed strongly on the <b>no tumor</b> class.
        <br><br>
        This demonstrates the trade-off between model
        performance and model complexity.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="custom-footer">NeuroVision AI • Model Evaluation Dashboard</div>',
        unsafe_allow_html=True
    )