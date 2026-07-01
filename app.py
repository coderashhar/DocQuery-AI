import os
import tempfile
import html as html_module
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

st.set_page_config(page_title="DocQuery AI", page_icon="📄", layout="wide")


# ═══════════════════════════════════════════════════════════════
#  CUSTOM CSS — Awwwards-style dark UI with animations
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ====== IMPORTS ====== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ====== ROOT VARIABLES ====== */
    :root {
        --primary-blue: #4285f4;
        --primary-purple: #a855f7;
        --primary-cyan: #06b6d4;
        --primary-pink: #ec4899;
        --glass-bg: rgba(15, 15, 30, 0.65);
        --glass-border: rgba(255, 255, 255, 0.08);
        --glass-border-hover: rgba(255, 255, 255, 0.16);
        --text-primary: #f0f0ff;
        --text-secondary: rgba(200, 200, 230, 0.7);
        --dark-bg: #06060e;
        --sidebar-bg: #0a0a16;
        --success-bg: rgba(34, 197, 94, 0.08);
        --success-border: rgba(34, 197, 94, 0.25);
        --error-bg: rgba(239, 68, 68, 0.08);
        --error-border: rgba(239, 68, 68, 0.25);
    }

    /* ====== ANIMATED AURORA BACKGROUND ====== */
    .stApp {
        background: var(--dark-bg) !important;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(ellipse 80% 50% at 20% 20%, rgba(66, 133, 244, 0.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 30%, rgba(168, 85, 247, 0.10) 0%, transparent 55%),
            radial-gradient(ellipse 70% 50% at 50% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 60%),
            radial-gradient(ellipse 50% 30% at 70% 60%, rgba(236, 72, 153, 0.06) 0%, transparent 50%);
        animation: auroraShift 14s ease-in-out infinite alternate;
    }

    @keyframes auroraShift {
        0% {
            background:
                radial-gradient(ellipse 80% 50% at 20% 20%, rgba(66, 133, 244, 0.12) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 30%, rgba(168, 85, 247, 0.10) 0%, transparent 55%),
                radial-gradient(ellipse 70% 50% at 50% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 60%),
                radial-gradient(ellipse 50% 30% at 70% 60%, rgba(236, 72, 153, 0.06) 0%, transparent 50%);
        }
        50% {
            background:
                radial-gradient(ellipse 70% 45% at 60% 15%, rgba(168, 85, 247, 0.14) 0%, transparent 60%),
                radial-gradient(ellipse 65% 50% at 25% 50%, rgba(6, 182, 212, 0.11) 0%, transparent 55%),
                radial-gradient(ellipse 80% 40% at 75% 75%, rgba(66, 133, 244, 0.09) 0%, transparent 60%),
                radial-gradient(ellipse 55% 35% at 40% 30%, rgba(236, 72, 153, 0.08) 0%, transparent 50%);
        }
        100% {
            background:
                radial-gradient(ellipse 85% 50% at 50% 30%, rgba(66, 133, 244, 0.13) 0%, transparent 60%),
                radial-gradient(ellipse 55% 45% at 30% 60%, rgba(168, 85, 247, 0.11) 0%, transparent 55%),
                radial-gradient(ellipse 70% 40% at 80% 50%, rgba(6, 182, 212, 0.10) 0%, transparent 60%),
                radial-gradient(ellipse 60% 35% at 20% 80%, rgba(236, 72, 153, 0.07) 0%, transparent 50%);
        }
    }

    /* ====== FLOATING PARTICLES ====== */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: 0;
        pointer-events: none;
        background-image:
            radial-gradient(1.5px 1.5px at 10% 15%, rgba(168, 85, 247, 0.45), transparent),
            radial-gradient(1.5px 1.5px at 20% 35%, rgba(66, 133, 244, 0.35), transparent),
            radial-gradient(1px 1px at 35% 55%, rgba(6, 182, 212, 0.4), transparent),
            radial-gradient(1.5px 1.5px at 45% 10%, rgba(236, 72, 153, 0.3), transparent),
            radial-gradient(1px 1px at 55% 75%, rgba(168, 85, 247, 0.35), transparent),
            radial-gradient(1.5px 1.5px at 65% 25%, rgba(6, 182, 212, 0.3), transparent),
            radial-gradient(1px 1px at 75% 65%, rgba(66, 133, 244, 0.4), transparent),
            radial-gradient(1.5px 1.5px at 85% 45%, rgba(236, 72, 153, 0.35), transparent),
            radial-gradient(1px 1px at 90% 85%, rgba(168, 85, 247, 0.4), transparent),
            radial-gradient(1px 1px at 30% 80%, rgba(66, 133, 244, 0.3), transparent);
        animation: particleDrift 20s linear infinite;
    }

    @keyframes particleDrift {
        0%   { transform: translateY(0px) translateX(0px); }
        25%  { transform: translateY(-8px) translateX(5px); }
        50%  { transform: translateY(-12px) translateX(8px); }
        75%  { transform: translateY(-5px) translateX(3px); }
        100% { transform: translateY(0px) translateX(0px); }
    }

    /* ====== Z-INDEX LAYERING ====== */
    .stMainBlockContainer,
    [data-testid="stSidebar"],
    [data-testid="stBottom"] {
        position: relative;
        z-index: 1;
    }

    /* ====== HIDE STREAMLIT CHROME ====== */
    #MainMenu, footer, .stDeployButton { visibility: hidden; display: none; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ====== GLOBAL TYPOGRAPHY ====== */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* ====== PREVENT UNNECESSARY SCROLL ====== */
    .stApp, html, body { overflow-x: hidden !important; }


    /* ═══════════════════════════════════════
       HERO HEADER
    ═══════════════════════════════════════ */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        margin-bottom: 0.5rem;
        animation: heroFadeIn 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    @keyframes heroFadeIn {
        from { opacity: 0; transform: translateY(-15px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .hero-icon {
        font-size: 3rem;
        display: block;
        margin-bottom: 0.5rem;
        animation: iconFloat 4s ease-in-out infinite;
    }

    @keyframes iconFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #4285f4 0%, #a855f7 35%, #ec4899 65%, #06b6d4 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientText 6s ease-in-out infinite;
        margin: 0 0 0.4rem 0 !important;
        padding: 0 !important;
        letter-spacing: -0.03em;
        line-height: 1.2 !important;
        display: block;
    }

    @keyframes gradientText {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
        letter-spacing: 0.02em;
        margin-top: 0.2rem;
        line-height: 1.5;
        text-align: center !important;
        width: 100%;
    }

    .hero-divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-blue), var(--primary-purple), var(--primary-cyan));
        border-radius: 2px;
        margin: 1.2rem auto 0;
        animation: dividerPulse 3s ease-in-out infinite alternate;
    }

    @keyframes dividerPulse {
        0% { box-shadow: 0 0 8px rgba(66, 133, 244, 0.4); width: 80px; opacity: 0.8; }
        100% { box-shadow: 0 0 18px rgba(168, 85, 247, 0.5); width: 120px; opacity: 1; }
    }


    /* ═══════════════════════════════════════
       SIDEBAR — GLASSMORPHISM
    ═══════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 10, 22, 0.95), rgba(15, 15, 35, 0.92)) !important;
        border-right: 1px solid var(--glass-border) !important;
        backdrop-filter: blur(30px) !important;
        -webkit-backdrop-filter: blur(30px) !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: transparent !important;
    }

    /* Sidebar header */
    .sidebar-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
        border-bottom: 1px solid var(--glass-border);
        margin-bottom: 1.5rem;
    }

    .sidebar-header .sidebar-icon {
        font-size: 2rem;
        display: block;
        margin-bottom: 0.4rem;
    }

    .sidebar-header h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        letter-spacing: -0.01em;
    }

    .sidebar-header p {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin: 0.3rem 0 0 0;
    }

    /* Hide default sidebar header/labels */
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h1 {
        display: none !important;
    }

    /* File uploader styling */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: var(--glass-bg) !important;
        border: 1px dashed rgba(168, 85, 247, 0.25) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        backdrop-filter: blur(16px) !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        border-color: rgba(168, 85, 247, 0.45) !important;
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.08) !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, rgba(66, 133, 244, 0.2), rgba(168, 85, 247, 0.2)) !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, rgba(66, 133, 244, 0.3), rgba(168, 85, 247, 0.35)) !important;
        box-shadow: 0 4px 16px rgba(168, 85, 247, 0.15) !important;
        transform: translateY(-1px);
    }

    /* File info in sidebar */
    [data-testid="stSidebar"] .uploadedFile {
        background: rgba(15, 15, 30, 0.5) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
    }

    /* Clear session button */
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        background: rgba(239, 68, 68, 0.08) !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
        border-radius: 12px !important;
        color: rgba(239, 115, 115, 0.9) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.65rem 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(12px) !important;
        margin-top: 0.8rem !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(239, 68, 68, 0.15) !important;
        border-color: rgba(239, 68, 68, 0.4) !important;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.1) !important;
        transform: translateY(-2px) !important;
    }

    [data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
    }

    /* Document status card */
    .doc-status {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-top: 1.2rem;
        backdrop-filter: blur(16px);
    }

    .doc-status .status-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.85rem;
    }

    .doc-status .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .doc-status .status-dot.active {
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
        animation: dotPulse 2s ease-in-out infinite;
    }

    .doc-status .status-dot.inactive {
        background: rgba(200, 200, 230, 0.3);
    }

    @keyframes dotPulse {
        0%, 100% { box-shadow: 0 0 8px rgba(34, 197, 94, 0.5); }
        50% { box-shadow: 0 0 16px rgba(34, 197, 94, 0.8); }
    }

    .doc-status .status-label {
        color: var(--text-secondary);
    }

    .doc-status .file-name {
        color: var(--text-primary);
        font-weight: 500;
        font-size: 0.82rem;
        margin-top: 0.5rem;
        padding-left: 1.4rem;
        word-break: break-all;
    }


    /* ═══════════════════════════════════════
       HIDE DEFAULT CHAT — USE CUSTOM BUBBLES
    ═══════════════════════════════════════ */
    .stChatMessage { display: none !important; }

    /* ====== CUSTOM CHAT AREA ====== */
    .chat-area {
        display: flex;
        flex-direction: column;
        gap: 0.7rem;
        padding: 0.5rem 0 1rem;
        max-width: 860px;
        margin: 0 auto;
    }

    .chat-bubble {
        max-width: 78%;
        padding: 0.9rem 1.25rem;
        font-size: 0.95rem;
        line-height: 1.6;
        color: var(--text-primary);
        word-wrap: break-word;
        animation: bubbleIn 0.4s cubic-bezier(0.22, 1, 0.36, 1);
        transition: all 0.25s ease;
    }

    @keyframes bubbleIn {
        from { opacity: 0; transform: translateY(12px) scale(0.97); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* ---- User bubble ---- */
    .chat-bubble.user-bubble {
        background: rgba(12, 12, 24, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 20px 20px 4px 20px;
        align-self: flex-end;
        backdrop-filter: blur(12px);
    }

    .chat-bubble.user-bubble:hover {
        border-color: rgba(255, 255, 255, 0.14);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }

    .chat-bubble .bubble-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
        display: block;
    }

    .chat-bubble.user-bubble .bubble-label {
        color: rgba(66, 133, 244, 0.7);
        text-align: right;
    }

    /* ---- AI bubble ---- */
    .chat-bubble.ai-bubble {
        background: linear-gradient(135deg, rgba(55, 65, 160, 0.25), rgba(168, 85, 247, 0.15) 45%, rgba(6, 182, 212, 0.10));
        border: 1px solid rgba(168, 85, 247, 0.15);
        border-radius: 20px 20px 20px 4px;
        align-self: flex-start;
        backdrop-filter: blur(12px);
    }

    .chat-bubble.ai-bubble:hover {
        border-color: rgba(168, 85, 247, 0.30);
        box-shadow: 0 4px 28px rgba(168, 85, 247, 0.08);
    }

    .chat-bubble.ai-bubble .bubble-label {
        color: rgba(168, 85, 247, 0.7);
    }

    /* ---- Empty state ---- */
    .chat-empty {
        text-align: center;
        padding: 4rem 1.5rem;
        animation: fadeInUp 0.6s cubic-bezier(0.22, 1, 0.36, 1);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .chat-empty .empty-icon {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 1rem;
        opacity: 0.35;
        animation: iconFloat 4s ease-in-out infinite;
    }

    .chat-empty .empty-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.4rem 0;
        opacity: 0.7;
    }

    .chat-empty .empty-desc {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin: 0;
        max-width: 400px;
        margin-left: auto;
        margin-right: auto;
    }

    .chat-empty .feature-pills {
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1.5rem;
    }

    .chat-empty .pill {
        font-size: 0.75rem;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        color: var(--text-secondary);
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }

    .chat-empty .pill:hover {
        border-color: rgba(168, 85, 247, 0.3);
        color: var(--text-primary);
    }


    /* ═══════════════════════════════════════
       CHAT INPUT BAR
    ═══════════════════════════════════════ */
    [data-testid="stBottom"] {
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stBottom"] > div {
        background: transparent !important;
    }

    .stChatInput {
        border-radius: 18px !important;
        overflow: hidden;
    }

    .stChatInput > div {
        background: rgba(12, 12, 28, 0.75) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 18px !important;
        backdrop-filter: blur(28px) !important;
        -webkit-backdrop-filter: blur(28px) !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stChatInput > div:focus-within {
        border-color: rgba(168, 85, 247, 0.4) !important;
        box-shadow:
            0 0 24px rgba(168, 85, 247, 0.10),
            0 0 48px rgba(66, 133, 244, 0.04) !important;
    }

    .stChatInput textarea {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }

    .stChatInput textarea::placeholder {
        color: var(--text-secondary) !important;
    }

    /* Send button */
    .stChatInput button {
        background: linear-gradient(135deg, var(--primary-blue), var(--primary-purple)) !important;
        border: none !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stChatInput button:hover {
        box-shadow: 0 0 22px rgba(168, 85, 247, 0.35) !important;
        transform: scale(1.08) !important;
    }

    .stChatInput button:active {
        transform: scale(0.96) !important;
    }


    /* ═══════════════════════════════════════
       ALERTS — SUCCESS / ERROR / WARNING
    ═══════════════════════════════════════ */
    /* Success */
    [data-testid="stSidebar"] div[data-testid="stAlert"][data-baseweb] {
        border-radius: 12px !important;
        backdrop-filter: blur(16px) !important;
    }

    .stAlert {
        border-radius: 12px !important;
        backdrop-filter: blur(16px) !important;
        animation: alertSlide 0.4s cubic-bezier(0.22, 1, 0.36, 1);
    }

    @keyframes alertSlide {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ═══════════════════════════════════════
       SPINNER
    ═══════════════════════════════════════ */
    .stSpinner > div {
        border-top-color: var(--primary-purple) !important;
    }

    /* ═══════════════════════════════════════
       SCROLLBAR
    ═══════════════════════════════════════ */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(168, 85, 247, 0.2);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(168, 85, 247, 0.4);
    }

    /* ═══════════════════════════════════════
       RESPONSIVE
    ═══════════════════════════════════════ */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.2rem !important; }
        .hero-subtitle { font-size: 0.88rem; }
        .chat-bubble { max-width: 90%; }
        .chat-empty .feature-pills { flex-direction: column; align-items: center; }
    }

    /* ═══════════════════════════════════════
       MAIN CONTAINER CENTERING
    ═══════════════════════════════════════ */
    .stMainBlockContainer > div {
        max-width: 900px;
        margin: 0 auto;
        padding-bottom: 5rem !important;
    }

    /* ═══════════════════════════════════════
       HIDE DEFAULT TITLE / DESCRIPTION
    ═══════════════════════════════════════ */
    .stMainBlockContainer h1 { display: none !important; }
    .stMainBlockContainer > div > div > div > .stMarkdown:first-child { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HERO HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <span class="hero-icon">📄</span>
    <span class="hero-title">DocQuery AI</span>
    <p class="hero-subtitle">Upload a document. Ask anything. Get intelligent, context-aware answers in seconds.</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR — Document Upload
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # Custom sidebar header
    st.markdown("""
    <div class="sidebar-header">
        <span class="sidebar-icon">🗂️</span>
        <h3>Document Hub</h3>
        <p>Upload & manage your PDFs</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], label_visibility="collapsed")

    if st.button("✕  Clear Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.current_file = None
        st.rerun()

    # Document status indicator
    if st.session_state.current_file:
        st.markdown(f"""
        <div class="doc-status">
            <div class="status-row">
                <span class="status-dot active"></span>
                <span class="status-label">Document loaded</span>
            </div>
            <div class="file-name">📎 {html_module.escape(st.session_state.current_file)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="doc-status">
            <div class="status-row">
                <span class="status-dot inactive"></span>
                <span class="status-label">No document loaded</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        # If a new file is uploaded, process it
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []  # Clear chat history

            with st.spinner("Processing document..."):
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    # Load and split
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs)

                    # Create embeddings and vectorstore (in-memory)
                    embedding_model = HuggingFaceEmbeddings()
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embedding_model
                    )

                    # Create retriever
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 3,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )
                    st.success("✓ Document processed — ready for questions!")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                finally:
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)


# ═══════════════════════════════════════════════════════════════
#  CHAT DISPLAY — Custom HTML Bubbles
# ═══════════════════════════════════════════════════════════════
chat_html_parts = []
has_chat = False

for message in st.session_state.messages:
    escaped = html_module.escape(message["content"])
    if message["role"] == "user":
        has_chat = True
        chat_html_parts.append(
            f'<div class="chat-bubble user-bubble">'
            f'<span class="bubble-label">You</span>'
            f'{escaped}</div>'
        )
    elif message["role"] == "assistant":
        has_chat = True
        chat_html_parts.append(
            f'<div class="chat-bubble ai-bubble">'
            f'<span class="bubble-label">DocQuery AI</span>'
            f'{escaped}</div>'
        )

if has_chat:
    st.markdown(
        '<div class="chat-area">' + ''.join(chat_html_parts) + '</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown("""
    <div class="chat-empty">
        <span class="empty-icon">✨</span>
        <p class="empty-title">Ready to explore your document</p>
        <p class="empty-desc">Upload a PDF from the sidebar, then ask any question — I'll find the answers.</p>
        <div class="feature-pills">
            <span class="pill">📑 PDF Analysis</span>
            <span class="pill">🔍 Smart Search</span>
            <span class="pill">💡 Context-Aware</span>
            <span class="pill">⚡ Instant Answers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CHAT INPUT & RAG LOGIC (unchanged)
# ═══════════════════════════════════════════════════════════════
if query := st.chat_input("Ask a question about your document..."):
    if not st.session_state.retriever:
        st.warning("⚠ Please upload a document first from the sidebar.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Setup LLM and Prompt
        llm = ChatMistralAI(model="mistral-small-latest")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a helpful assistant that provides concise answers based on retrieved documents. If the retrieved documents do not contain enough information to answer the question, say "I could not find the information in the given document" """),
                ("human", "Context:{context}\n\nQuestion:{question}")
            ]
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve documents
                    docs = st.session_state.retriever.invoke(query)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Generate response
                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })

                    response = llm.invoke(final_prompt)
                    answer = response.content

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred during response generation: {e}")

        st.rerun()
