# 📚 MBU Study Vault – AI Powered Notes & Question Paper Repository

A mini project (MCA 3rd Semester) built with **Python, Streamlit, SQLite and NLTK**.

MBU Study Vault lets Mohan Babu University students **upload, search, view and
download** Handwritten Notes, Typed Notes and Previous Question Papers —
organized by Course → Semester → Subject — instead of searching endlessly
through WhatsApp/Telegram groups.

---

## 🧠 What makes it "AI Powered"?

The search feature uses a small **Natural Language Processing (NLP) pipeline**
built with **NLTK**:

1. **Tokenization** – breaks the search query into individual words
2. **Stopword Removal** – removes common English words like "the", "is", "of"
3. **Keyword Matching** – compares the remaining keywords against keywords
   stored for every uploaded file, and ranks results by how many keywords match

This is intentionally kept simple (no heavy Machine Learning models) so it is
easy to explain during a viva/demonstration.

---

## 🗂 Project Structure

```
MBU_Study_Vault/
│
├── app.py              # Entire application (UI + DB + Auth + NLP search)
├── requirements.txt     # Python dependencies
├── uploads/              # Uploaded PDF files are stored here (auto-created)
└── studyvault.db         # SQLite database (auto-created on first run)
```

Everything (UI, database logic, authentication, upload handling and search)
is written inside a single `app.py` file on purpose — to keep the project
simple and beginner-friendly for viva/demonstration.

---

## ⚙️ Technology Used

| Layer          | Technology         |
|----------------|---------------------|
| Language       | Python 3            |
| Web UI         | Streamlit           |
| Database       | SQLite (built into Python) |
| NLP            | NLTK (Tokenization + Stopwords) |

No Java, Eclipse, Django, React or Angular is used anywhere in this project.

---

## ▶️ How to Run (VS Code)

1. Open the `MBU_Study_Vault` folder in VS Code.
2. Open a terminal inside VS Code (`Ctrl + ~`).
3. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```
6. The app will **automatically open in your default browser** at
   `http://localhost:8501`.

That's it! The SQLite database (`studyvault.db`) and the `uploads/` folder
are created automatically the first time you run the app — no manual setup
needed.

---

## 🖼 Adding the MBU Logo

Right now the Home page and Sidebar show a gold placeholder badge with the
text "MBU". Once you have the official Mohan Babu University logo:

1. Rename the image file to exactly `mbu_logo.png`
2. Place it in the same folder as `app.py`
3. Restart the app — the logo will automatically replace the placeholder.

---

## 👨‍🎓 How to Use

1. **Signup** with your name, email and password.
2. **Login** with your credentials.
3. Go to **Dashboard** → select **Course** → **Semester** → **Subject** to
   browse existing material.
4. Click **➕ Upload Material** to add your own Notes / Question Papers
   (PDF only).
5. Use **🔍 Search** and type a keyword like `Python`, `DBMS`,
   `Cloud Computing` or `Data Mining` to instantly find related material.
6. Click **⬇ Download** on any file to save it.

---

## 🎨 UI Theme

Dark Maroon, Gold, White and Light Grey — rounded cards, modern buttons and
clean spacing, designed to look premium and professional (no blue, since it
was already used in a previous mini project).

---

## 🚀 Future Enhancements

These are natural next steps if this project is extended beyond the mini
project scope:

- **OCR for Handwritten Notes** – extract text from scanned handwritten PDFs
  so they become searchable too.
- **AI Summary** – auto-generate short summaries of long notes.
- **AI Important Questions** – automatically highlight frequently repeated
  questions from previous question papers.
- **Admin Approval** – require admin approval before an uploaded file
  becomes publicly visible.
- **Ratings** – let students rate the usefulness of a note/paper.
- **Comments** – allow students to discuss or ask doubts on a file.
- **Notifications** – notify students when new material is uploaded for
  their subjects.

---

## 📄 License

Built for academic/educational purposes as an MCA mini project submission.
