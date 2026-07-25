"""
MBU Study Vault - AI Powered Notes & Question Paper Repository
----------------------------------------------------------------
A beginner-friendly mini project built with Streamlit + Supabase + NLTK.

Run with:
    streamlit run app.py

Everything (UI, database, authentication, upload, NLP search) lives
in this single file on purpose, to keep the project simple.

Environment variables required (set these before running):
    SUPABASE_URL   -> your Supabase project URL
    SUPABASE_KEY   -> your Supabase service_role or anon API key

Expected Supabase setup (create these once in your Supabase project):

    Table: users
        id             bigint, primary key, identity
        full_name      text, not null
        email          text, unique, not null
        password_hash  text, not null
        created_at     text, not null

    Table: files
        id               bigint, primary key, identity
        course           text, not null
        semester         text, not null
        subject          text, not null
        category         text, not null
        display_name     text, not null
        stored_filename  text, not null   (path inside the storage bucket)
        keywords         text
        uploaded_by      text
        upload_date      text, not null

    Storage bucket: uploads
        A bucket named "uploads" (can be public or private - the app
        downloads files through the Supabase SDK either way).
"""

import os
import re
import uuid
import hashlib
from datetime import datetime

import streamlit as st
import nltk
from supabase import create_client

# ======================================================================
# 1. PAGE CONFIG (must be the very first Streamlit command)
# ======================================================================
st.set_page_config(
    page_title="MBU Study Vault",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# 2. CONSTANTS & PATHS
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NOTE: Drop your official MBU logo file (named exactly "mbu_logo.png")
# into this same folder later. Until then, a gold placeholder badge is shown.
LOGO_PATH = os.path.join(BASE_DIR, "mbu_logo.png")

STORAGE_BUCKET = "uploads"

# The single administrator account. Logging in with this email unlocks
# file-delete permissions in render_file_list().
ADMIN_EMAIL = "varshaseenu2005@gmail.com"

COURSES = {
    "MCA": [f"Semester {i}" for i in range(1, 5)],
    "BCA": [f"Semester {i}" for i in range(1, 7)],
    "B.Tech": [f"Semester {i}" for i in range(1, 9)],
    "MBA": [f"Semester {i}" for i in range(1, 5)],
}

CATEGORIES = ["Handwritten Notes", "Typed Notes", "Question Paper"]


# ======================================================================
# 3. SIMPLE NLP SETUP (Tokenization + Stopword Removal)
#    Uses NLTK. Required data packages are downloaded automatically
#    the first time the app runs (only if they are missing).
# ======================================================================
@st.cache_resource
def setup_nltk():
    required_packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for find_path, package_name in required_packages:
        try:
            nltk.data.find(find_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                # If there is no internet the app still works using the
                # basic fallback tokenizer defined in get_keywords().
                pass
    return True


setup_nltk()

try:
    from nltk.corpus import stopwords as nltk_stopwords
    STOPWORDS = set(nltk_stopwords.words("english"))
except Exception:
    STOPWORDS = set()

try:
    from nltk.tokenize import word_tokenize
except Exception:
    word_tokenize = None


def get_keywords(text):
    """
    A very small NLP pipeline used for both indexing uploaded files
    and understanding search queries:

        1. Lowercase the text
        2. Remove punctuation/special characters (keep + and # so that
           things like "C++" or "C#" are not destroyed)
        3. Tokenize the text into individual words
        4. Remove common English stopwords ("the", "is", "of", ...)
        5. Remove very short / empty tokens

    Returns a list of clean, unique keywords.
    """
    if not text:
        return []

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)

    if word_tokenize is not None:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()

    keywords = []
    seen = set()
    for token in tokens:
        token = token.strip()
        if token and token not in STOPWORDS and len(token) > 1 and token not in seen:
            seen.add(token)
            keywords.append(token)

    return keywords


# ======================================================================
# 4. SUPABASE CLIENT SETUP
# ======================================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


@st.cache_resource
def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error(
            "Missing Supabase configuration. Please set the SUPABASE_URL and "
            "SUPABASE_KEY environment variables before running the app."
        )
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_KEY)


supabase = init_supabase()


# ======================================================================
# 5. AUTHENTICATION HELPERS
# ======================================================================
def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(full_name, email, password):
    """Insert a new user into the Supabase `users` table."""
    email = email.lower().strip()
    try:
        existing = (
            supabase.table("users")
            .select("id")
            .eq("email", email)
            .execute()
        )
        if existing.data:
            return False, "An account with this email already exists."

        supabase.table("users").insert(
            {
                "full_name": full_name.strip(),
                "email": email,
                "password_hash": hash_password(password),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ).execute()
        return True, "Account created successfully! Please login."
    except Exception as e:
        return False, f"Could not create account: {e}"


def verify_user(email, password):
    """Verify email + hashed password against the Supabase `users` table."""
    email = email.lower().strip()
    try:
        response = (
            supabase.table("users")
            .select("*")
            .eq("email", email)
            .execute()
        )
        if response.data:
            user = response.data[0]
            if user.get("password_hash") == hash_password(password):
                return user
    except Exception:
        pass
    return None


# ======================================================================
# 6. FILE / SEARCH HELPERS (Supabase table `files` + Storage bucket)
# ======================================================================
def save_file_record(course, semester, subject, category, display_name, stored_filename, uploaded_by):
    keyword_source = f"{course} {semester} {subject} {category} {display_name}"
    keywords = " ".join(get_keywords(keyword_source))

    supabase.table("files").insert(
        {
            "course": course,
            "semester": semester,
            "subject": subject,
            "category": category,
            "display_name": display_name,
            "stored_filename": stored_filename,
            "keywords": keywords,
            "uploaded_by": uploaded_by,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    ).execute()


def get_subjects(course, semester):
    response = (
        supabase.table("files")
        .select("subject")
        .eq("course", course)
        .eq("semester", semester)
        .execute()
    )
    subjects = sorted({row["subject"] for row in response.data if row.get("subject")})
    return subjects


def get_files(course=None, semester=None, subject=None):
    query = supabase.table("files").select("*")
    if course:
        query = query.eq("course", course)
    if semester:
        query = query.eq("semester", semester)
    if subject:
        query = query.eq("subject", subject)
    response = query.order("upload_date", desc=True).execute()
    return response.data or []


def search_files(user_query):
    """
    NLP based search.

    1. Extract clean keywords from the student's search query.
    2. Compare those keywords against the pre-computed keyword list of
       every uploaded file (built at upload time in save_file_record).
    3. Files are ranked by how many keywords matched, so the most
       relevant results appear first.
    """
    query_keywords = get_keywords(user_query)
    if not query_keywords:
        return []

    response = supabase.table("files").select("*").execute()
    rows = response.data or []

    results = []
    for row in rows:
        file_keywords = set((row.get("keywords") or "").split())
        match_count = sum(1 for k in query_keywords if k in file_keywords)

        # Fallback: also allow simple substring matching so short forms
        # like "dbms" still match even if tokenization split things up.
        if match_count == 0:
            combined_text = (
                f"{row['course']} {row['semester']} {row['subject']} "
                f"{row['category']} {row['display_name']}"
            ).lower()
            match_count = sum(1 for k in query_keywords if k in combined_text)

        if match_count > 0:
            row["match_score"] = match_count
            results.append(row)

    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results


def upload_file_to_storage(uploaded_file):
    """
    Uploads the given Streamlit UploadedFile to the Supabase Storage
    bucket "uploads" and returns the storage path to save as
    `stored_filename` in the `files` table.
    """
    unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    file_bytes = uploaded_file.getvalue()

    supabase.storage.from_(STORAGE_BUCKET).upload(
        unique_name,
        file_bytes,
        {"content-type": "application/pdf"},
    )
    return unique_name


def download_file_from_storage(stored_filename):
    """
    Downloads the raw bytes of a file from the Supabase Storage bucket.
    Returns None if the file cannot be found/downloaded.
    """
    try:
        return supabase.storage.from_(STORAGE_BUCKET).download(stored_filename)
    except Exception:
        return None


def delete_file(file_id, stored_filename):
    """
    Admin-only action: removes the PDF from the Supabase Storage bucket
    and deletes the matching row from the `files` table.
    Caller (render_file_list) is responsible for checking admin rights.
    """
    try:
        supabase.storage.from_(STORAGE_BUCKET).remove([stored_filename])
        supabase.table("files").delete().eq("id", file_id).execute()
        return True, "File deleted successfully."
    except Exception as e:
        return False, f"Could not delete file: {e}"


# ======================================================================
# 7. STYLING - Dark Maroon / Gold / White / Light Grey theme
# ======================================================================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background-color: #F4F2EF;
        }

        section[data-testid="stSidebar"] {
            background-color: #4A0E0E;
        }
        section[data-testid="stSidebar"] * {
            color: #F5EFE0 !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
            background-color: #6B1F1F;
            color: #F5D57A !important;
            border: 1px solid #D4AF37;
            border-radius: 10px;
            font-weight: 600;
            margin-bottom: 6px;
            width: 100%;
            transition: all 0.2s ease-in-out;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background-color: #D4AF37;
            color: #4A0E0E !important;
            transform: translateY(-2px);
        }

        .logo-placeholder {
            width: 90px; height: 90px; border-radius: 50%;
            background: linear-gradient(135deg, #D4AF37, #F5D57A);
            color: #4A0E0E; font-weight: 700; font-size: 22px;
            display: flex; align-items: center; justify-content: center;
            margin: 10px auto; border: 3px solid #F5EFE0;
        }
        .logo-placeholder-big {
            width: 130px; height: 130px; border-radius: 50%;
            background: linear-gradient(135deg, #D4AF37, #F5D57A);
            color: #4A0E0E; font-weight: 700; font-size: 32px;
            display: flex; align-items: center; justify-content: center;
            margin: 10px auto; border: 4px solid #4A0E0E;
        }

        .hero { text-align: center; padding: 6px 0 0 0; }
        .hero-title {
            color: #4A0E0E; font-size: 42px; font-weight: 700;
            margin-bottom: 0px; text-align: center;
        }
        .hero-sub {
            color: #B8860B; font-size: 18px; font-weight: 600;
            text-align: center; margin-top: 0px;
        }

        .welcome-card {
            background: #FFFFFF; border-left: 6px solid #D4AF37;
            border-radius: 14px; padding: 22px 28px; margin: 20px 0;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08); color: #333333;
        }
        .welcome-card h3 { color: #4A0E0E; margin-top: 0; }

        .feature-card {
            background: #FFFFFF; border-radius: 16px; padding: 20px 14px;
            text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.07);
            transition: transform 0.2s ease-in-out; min-height: 175px;
            border-top: 4px solid #D4AF37;
        }
        .feature-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }
        .feature-icon { font-size: 30px; margin-bottom: 8px; }
        .feature-title { font-weight: 700; color: #4A0E0E; margin-bottom: 6px; }
        .feature-desc { font-size: 13px; color: #666666; }

        .course-card {
            background: linear-gradient(135deg, #4A0E0E, #6B1F1F);
            border-radius: 16px; padding: 26px 10px; text-align: center;
            color: #F5D57A; box-shadow: 0 4px 14px rgba(0,0,0,0.15);
            margin-bottom: 6px; font-weight: 700;
        }
        .course-icon { font-size: 26px; }
        .course-title { font-size: 18px; margin-top: 6px; }

        .subject-card {
            background: #FFFFFF; border-radius: 12px; padding: 14px;
            text-align: center; font-weight: 600; color: #4A0E0E;
            border: 1px solid #E8E0D0; margin-bottom: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        .file-row { padding: 8px 0; }
        .file-name { font-weight: 600; color: #4A0E0E; font-size: 16px; }
        .file-meta { color: #777777; font-size: 13px; margin-top: 2px; }
        .badge {
            background: #D4AF37; color: #4A0E0E; padding: 2px 10px;
            border-radius: 20px; font-size: 11px; font-weight: 700;
        }
        .upload-date { color: #999999; font-size: 12px; padding-top: 14px; }
        .file-divider { border: none; border-top: 1px solid #E4E0D8; margin: 4px 0 14px 0; }

        div.stButton > button:first-child {
            background-color: #4A0E0E; color: #F5D57A;
            border: 1px solid #D4AF37; border-radius: 10px; font-weight: 600;
        }
        div.stButton > button:first-child:hover {
            background-color: #D4AF37; color: #4A0E0E; border: 1px solid #4A0E0E;
        }

        .stDownloadButton > button {
            background-color: #D4AF37 !important; color: #4A0E0E !important;
            border: 1px solid #4A0E0E !important; border-radius: 10px; font-weight: 700;
        }
        .stDownloadButton > button:hover {
            background-color: #4A0E0E !important; color: #F5D57A !important;
        }

        .footer {
            text-align: center; color: #999999; font-size: 12px;
            padding: 30px 0 10px 0;
        }

        h1, h2, h3 { color: #4A0E0E; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================
# 8. SESSION STATE DEFAULTS
# ======================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "selected_course" not in st.session_state:
    st.session_state.selected_course = None
if "selected_semester" not in st.session_state:
    st.session_state.selected_semester = None
if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = None


# ======================================================================
# 9. SIDEBAR NAVIGATION
# ======================================================================
def sidebar_nav():
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown('<div class="logo-placeholder">MBU</div>', unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align:center;color:#D4AF37;'>Study Vault</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if st.session_state.logged_in:
            first_name = st.session_state.user["full_name"].split()[0]
            st.markdown(f"**Welcome, {first_name}!**")

            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = "Home"
                st.rerun()
            if st.button("📂 Dashboard", use_container_width=True):
                st.session_state.page = "Dashboard"
                st.rerun()
            if st.button("➕ Upload Material", use_container_width=True):
                st.session_state.page = "Upload"
                st.rerun()
            if st.button("🔍 Search", use_container_width=True):
                st.session_state.page = "Search"
                st.rerun()

            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.session_state.page = "Home"
                st.rerun()
        else:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = "Home"
                st.rerun()
            if st.button("📂 Dashboard", use_container_width=True):
                st.session_state.page = "Dashboard"
                st.rerun()
            if st.button("🔍 Search", use_container_width=True):
                st.session_state.page = "Search"
                st.rerun()

            st.markdown("---")
            if st.button("🔐 Login", use_container_width=True):
                st.session_state.page = "Login"
                st.rerun()
            if st.button("📝 Signup", use_container_width=True):
                st.session_state.page = "Signup"
                st.rerun()


# ======================================================================
# 10. SHARED UI COMPONENT - file list with download buttons
# ======================================================================
def render_file_list(files):
    if not files:
        st.info("No files found.")
        return

    is_admin = (
        st.session_state.logged_in
        and st.session_state.user
        and st.session_state.user.get("email", "").lower().strip() == ADMIN_EMAIL
    )

    for f in files:
        if is_admin:
            c1, c2, c3, c4 = st.columns([5, 2, 2, 1])
        else:
            c1, c2, c3 = st.columns([5, 2, 2])

        with c1:
            st.markdown(
                f"""
                <div class="file-row">
                    <div class="file-name">📄 {f['display_name']}</div>
                    <div class="file-meta">{f['course']} • {f['semester']} • {f['subject']} •
                        <span class="badge">{f['category']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='upload-date'>Uploaded: {f['upload_date'][:10]}</div>",
                unsafe_allow_html=True,
            )
        with c3:
            file_bytes = download_file_from_storage(f["stored_filename"])
            if file_bytes:
                download_name = f["display_name"]
                if not download_name.lower().endswith(".pdf"):
                    download_name += ".pdf"
                st.download_button(
                    label="⬇ Download",
                    data=file_bytes,
                    file_name=download_name,
                    mime="application/pdf",
                    key=f"download_{f['id']}",
                    use_container_width=True,
                )
            else:
                st.error("File missing on server")

        if is_admin:
            with c4:
                if st.button("🗑", key=f"delete_{f['id']}", use_container_width=True):
                    success, message = delete_file(f["id"], f["stored_filename"])
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        st.markdown("<hr class='file-divider'>", unsafe_allow_html=True)


# ======================================================================
# 11. PAGES
# ======================================================================
def render_home():
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=140)
        else:
            st.markdown('<div class="logo-placeholder-big">MBU</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">MBU Study Vault</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-sub">AI Powered Notes &amp; Question Paper Repository</p>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="welcome-card">
        <h3>Welcome to MBU Study Vault 👋</h3>
        <p>
        No more endless scrolling through WhatsApp and Telegram groups looking for notes.
        MBU Study Vault is a single place where Mohan Babu University students can
        <b>upload</b>, <b>search</b>, <b>view</b> and <b>download</b> handwritten notes,
        typed notes and previous question papers — organized by Course, Semester and
        Subject, and powered by a simple built-in AI search engine.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Why MBU Study Vault?")
    features = [
        ("📚", "Organized Material", "Notes and papers sorted by Course, Semester & Subject."),
        ("🔍", "Smart NLP Search", "Type a keyword like 'Python' or 'DBMS' and find everything instantly."),
        ("⬆️", "Easy Uploads", "Upload PDFs in seconds and help future students."),
        ("⬇️", "Free Downloads", "Download any material with a single click."),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    if not st.session_state.logged_in:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Get Started → Login / Signup", use_container_width=True):
                st.session_state.page = "Login"
                st.rerun()


def render_signup():
    st.markdown("## 📝 Student Signup")
    with st.form("signup_form"):
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account")

        if submitted:
            if not full_name or not email or not password:
                st.error("Please fill all the fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 4:
                st.error("Password should be at least 4 characters long.")
            else:
                success, message = create_user(full_name, email, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.markdown("Already have an account?")
    if st.button("Go to Login"):
        st.session_state.page = "Login"
        st.rerun()


def render_login():
    st.markdown("## 🔐 Student Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            user = verify_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.session_state.page = "Dashboard"
                st.success(f"Welcome back, {user['full_name']}!")
                st.rerun()
            else:
                st.error("Invalid email or password.")

    st.markdown("Don't have an account?")
    if st.button("Go to Signup"):
        st.session_state.page = "Signup"
        st.rerun()


def render_dashboard():
    st.markdown("## 📂 Dashboard")
    st.markdown("Select your **Course** to get started.")

    course_cols = st.columns(len(COURSES))
    for col, course in zip(course_cols, COURSES.keys()):
        with col:
            st.markdown(
                f"""
                <div class="course-card">
                    <div class="course-icon">🎓</div>
                    <div class="course-title">{course}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Select {course}", key=f"course_{course}", use_container_width=True):
                st.session_state.selected_course = course
                st.session_state.selected_semester = None
                st.session_state.selected_subject = None
                st.rerun()

    if st.session_state.selected_course:
        st.markdown("---")
        st.markdown(f"### {st.session_state.selected_course} — Select Semester")
        semesters = COURSES[st.session_state.selected_course]
        sem_cols = st.columns(4)
        for i, sem in enumerate(semesters):
            with sem_cols[i % 4]:
                if st.button(sem, key=f"sem_{sem}", use_container_width=True):
                    st.session_state.selected_semester = sem
                    st.session_state.selected_subject = None
                    st.rerun()

    if st.session_state.selected_course and st.session_state.selected_semester:
        st.markdown("---")
        st.markdown(
            f"### {st.session_state.selected_course} — "
            f"{st.session_state.selected_semester} — Subjects"
        )
        subjects = get_subjects(st.session_state.selected_course, st.session_state.selected_semester)

        if not subjects:
            st.info("No subjects uploaded yet for this semester. Be the first to upload!")
            if st.button("➕ Upload Material for this Semester"):
                st.session_state.page = "Upload"
                st.rerun()
        else:
            subj_cols = st.columns(3)
            for i, subject in enumerate(subjects):
                with subj_cols[i % 3]:
                    st.markdown(f'<div class="subject-card">📘 {subject}</div>', unsafe_allow_html=True)
                    if st.button(f"View {subject}", key=f"subj_{subject}", use_container_width=True):
                        st.session_state.selected_subject = subject
                        st.rerun()

    if (
        st.session_state.selected_course
        and st.session_state.selected_semester
        and st.session_state.selected_subject
    ):
        st.markdown("---")
        st.markdown(f"### 📄 Files — {st.session_state.selected_subject}")
        files = get_files(
            st.session_state.selected_course,
            st.session_state.selected_semester,
            st.session_state.selected_subject,
        )
        render_file_list(files)


def render_upload():
    st.markdown("## ➕ Upload Study Material")

    if not st.session_state.logged_in:
        st.warning("Please login to upload study material.")
        if st.button("Go to Login"):
            st.session_state.page = "Login"
            st.rerun()
        return

    with st.form("upload_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            course = st.selectbox("Course", list(COURSES.keys()))
        with col2:
            semester = st.selectbox("Semester", COURSES[course])

        subject = st.text_input("Subject (e.g. Python, DBMS, Cloud Computing)")
        category = st.selectbox("File Type", CATEGORIES)
        display_name = st.text_input("File Name (what should students see?)")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        submitted = st.form_submit_button("Upload")

        if submitted:
            if not subject or not display_name or uploaded_file is None:
                st.error("Please fill all fields and choose a PDF file.")
            else:
                try:
                    stored_filename = upload_file_to_storage(uploaded_file)
                    save_file_record(
                        course,
                        semester,
                        subject.strip(),
                        category,
                        display_name.strip(),
                        stored_filename,
                        st.session_state.user["full_name"],
                    )
                    st.success(
                        f"'{display_name}' uploaded successfully under "
                        f"{course} - {semester} - {subject}!"
                    )
                except Exception as e:
                    st.error(f"Upload failed: {e}")


def render_search():
    st.markdown("## 🔍 Search Study Material")
    st.markdown(
        "Powered by a simple built-in NLP engine "
        "(Tokenization + Stopword Removal + Keyword Matching)."
    )

    query = st.text_input(
        "Search by keyword — e.g. Python, Java, DBMS, Cloud Computing, Data Mining"
    )
    search_clicked = st.button("Search")

    if search_clicked or query:
        if query and query.strip():
            results = search_files(query)
            st.markdown(f"**{len(results)} result(s) found for** \"{query}\"")
            render_file_list(results)
        else:
            st.info("Type a keyword to search.")


# ======================================================================
# 12. MAIN ROUTING
# ======================================================================
inject_css()
sidebar_nav()

current_page = st.session_state.page

if current_page == "Home":
    render_home()
elif current_page == "Signup":
    render_signup()
elif current_page == "Login":
    render_login()
elif current_page == "Dashboard":
    render_dashboard()
elif current_page == "Upload":
    render_upload()
elif current_page == "Search":
    render_search()
else:
    render_home()

st.markdown(
    """
    <div class="footer">
        Made with ❤️ for Mohan Babu University Students | MBU Study Vault © 2026
    </div>
    """,
    unsafe_allow_html=True,
)
