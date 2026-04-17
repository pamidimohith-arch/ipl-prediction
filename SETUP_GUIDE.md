## IPL PREDICTION 2026 - SETUP GUIDE

### 📋 Prerequisites
- Python 3.8 or higher installed
- Terminal/Command Prompt access
- Text Editor (VS Code recommended)

---

## Step-by-Step Setup Instructions

### 1️⃣ Navigate to Project Folder

```bash
cd "/Users/pamidimohith/python projects/IPL PREDICTION 2026"
```

### 2️⃣ Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
```

**On Windows:**
```bash
python -m venv venv
```

### 3️⃣ Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

✅ **You'll see `(venv)` at the start of your terminal line when activated**

### 4️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5️⃣ Verify Installation

```bash
python -c "import pandas, numpy, sklearn, streamlit; print('All libraries installed! ✅')"
```

---

## What Each Library Does

| Library | Purpose |
|---------|---------|
| **pandas** | Read CSV files and manipulate data |
| **numpy** | Numerical computations |
| **scikit-learn** | Machine Learning algorithms |
| **joblib** | Save/load trained models |
| **streamlit** | Create web UI (optional) |
| **matplotlib** | Plot charts and graphs |
| **seaborn** | Beautiful data visualization |

---

## Common Issues & Fixes

### ❌ Issue: "python3: command not found"
**Fix:** Install Python from python.org or use homebrew:
```bash
brew install python3
```

### ❌ Issue: "Permission denied" when activating venv
**Fix:** Try:
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

### ❌ Issue: "No module named pandas"
**Fix:** Make sure venv is activated and reinstall:
```bash
pip install -r requirements.txt
```

### ❌ Issue: "ModuleNotFoundError: No module named 'src'"
**Fix:** Run `main.py` from the project root directory

---

## Directory Structure Check

After setup, your folder should look like:
```
IPL PREDICTION 2026/
├── venv/                          ← Virtual environment (auto-created)
├── data/
├── src/
├── models/
├── requirements.txt               ← Dependencies list
├── .gitignore                     ← GitHub ignore file
└── SETUP_GUIDE.md                ← This file
```

---

## You're Ready! 🚀

Once setup is complete:
1. Keep the virtual environment activated
2. Proceed to STEP 2: Data Loading
3. Create `main.py` file
4. Run the code!
