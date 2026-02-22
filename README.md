# 🚦 Traffic Sign Recognition AI — Setup Guide

Welcome! This guide will walk you through everything step by step.  
Even if you're a complete beginner, just follow each step in order.

---

## 📋 What You Need First

- **Python 3.9 or newer** installed on your computer
  - Download from: https://www.python.org/downloads/
  - ✅ During installation, check **"Add Python to PATH"**
- A **dataset** of traffic sign images (see below)

---

## 📁 Step 1 — Organize Your Dataset

Your images must be organized in folders like this:

```
dataset/
  ├── stop_sign/
  │     ├── img1.jpg
  │     ├── img2.jpg
  │     └── img3.jpg
  ├── speed_limit_30/
  │     ├── img1.jpg
  │     └── img2.jpg
  └── yield/
        ├── img1.jpg
        └── img2.jpg
```

**Rules:**
- The `dataset` folder must be in the **same folder as train_model.py**
- Each **subfolder** = one type of sign
- Folder names become the label (e.g. `stop_sign` → "Stop Sign")
- Supports: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Aim for at least **30–50 images per category** for decent accuracy

---

## 💻 Step 2 — Install Required Libraries

Open a **terminal** (Command Prompt on Windows, Terminal on Mac/Linux).

Navigate to your project folder:
```
cd path/to/your/project/folder
```

Then run this command to install everything:
```
pip install tensorflow pillow numpy scikit-learn flet
```

> ⏳ This may take a few minutes — TensorFlow is a large package.

---

## 🧠 Step 3 — Train the AI Model

In your terminal, run:
```
python train_model.py
```

You'll see output like:
```
✅ Found 3 categories: ['stop_sign', 'speed_limit', 'yield']
   Loaded 150 images total.
🚀 Training for 15 epochs...
   Epoch 1/15 ...
   Test Accuracy: 89.33%
✅ Model saved as: traffic_sign_model.h5
```

This creates two files:
- `traffic_sign_model.h5` — the trained AI brain
- `class_names.txt` — list of sign categories

> 💡 **Tip:** More training images = better accuracy!

---

## 🖥️ Step 4 — Launch the App

Run the UI app:
```
python app.py
```

A window will open. You can:
1. Click **"Choose Image"** to upload a traffic sign photo
2. Click **"Analyze Sign"** to run the AI
3. See the result with confidence score and top 3 predictions!

---

## ❓ Common Problems & Fixes

| Problem | Fix |
|---|---|
| `pip` not found | Reinstall Python and check "Add to PATH" |
| `ModuleNotFoundError: tensorflow` | Run `pip install tensorflow` again |
| Model file not found | Run `train_model.py` before `app.py` |
| Low accuracy (e.g. 50%) | Add more images to your dataset |
| App window doesn't open | Make sure Flet installed: `pip install flet` |

---

## 🧪 Tips to Improve Accuracy

- **More data** — aim for 100+ images per category
- **Diverse images** — different angles, lighting, distances
- **More epochs** — open `train_model.py`, change `EPOCHS = 15` to `EPOCHS = 30`
- **Clean data** — remove blurry or mislabeled images

---

## 📂 Final File Structure

```
your-project/
  ├── dataset/          ← your image folders
  ├── train_model.py    ← run this FIRST to train
  ├── app.py            ← run this to use the app
  ├── traffic_sign_model.h5   ← created after training
  ├── class_names.txt         ← created after training
  └── README.md         ← this file
```

---

Good luck with your project! 🎉
