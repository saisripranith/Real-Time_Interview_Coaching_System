# 🚀 Interview Emotion Analysis - Setup Guide

A complete guide to get this project running from scratch on any Windows/Mac/Linux machine.

---

## 📋 Prerequisites

Your friend needs to install these first (if not already present):

### 1. **Python** (for backend)
- Download: https://www.python.org/downloads/
- **Important:** During installation, check "Add Python to PATH"
- Verify: Open terminal/cmd and run `python --version`

### 2. **Node.js & npm/bun** (for frontend)
- Download: https://nodejs.org/
- Choose **LTS (Long Term Support)** version
- Verify: Open terminal/cmd and run `node --version` and `npm --version`

### 3. **Git** (to clone the repo)
- Download: https://git-scm.com/
- Verify: Open terminal/cmd and run `git --version`

### 4. **Firebase Setup** (Optional - for saving results online)
- Create a Firebase project at https://console.firebase.google.com/
- Get your Firebase config (we'll need this later)

---

## 🎯 Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Interview_Emotion_Analysis.git
cd Interview_Emotion_Analysis
```

---

### Step 2: Set Up Backend

```bash
# Navigate to backend folder
cd backend

# Create a virtual environment (Python)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# You're done with backend setup!
```

---

### Step 3: Set Up Frontend

```bash
# Navigate to frontend folder (from root directory)
cd frontend

# Install dependencies
npm install
# OR if using bun:
bun install
```

**Optional: Set up Firebase** (if you want to save results online)

Edit `frontend/src/lib/firebase.ts` and add your Firebase config:

```typescript
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_AUTH_DOMAIN",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_STORAGE_BUCKET",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID",
};
```

---

## ▶️ Running the Project

### Open **3 Terminal Windows** (or tabs):

#### **Terminal 1: Backend Server**
```bash
cd Interview_Emotion_Analysis/backend

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Run the backend
python main.py
```

You should see:
```
🚀 Starting Interview Coaching Backend...
📷 Initializing Head Pose Detector...
✅ Head Pose Detector ready!
🎤 Initializing Speech Analyzer...
✅ Speech Analyzer ready!
🎉 Backend fully initialized and ready!
```

Backend will run on `http://localhost:8000`

---

#### **Terminal 2: Frontend Dev Server**
```bash
cd Interview_Emotion_Analysis/frontend

# Using npm:
npm run dev

# OR using bun:
bun run dev
```

You should see:
```
VITE v... ready in ... ms

➜ Local:   http://localhost:5173/
➜ press h + enter to show help
```

---

#### **Terminal 3: Just Keep Open (For Logs)**

Keep this terminal open to watch for any error messages.

---

## 🌐 Access the App

Open your browser and go to:
```
http://localhost:5173
```

You should see the Interview Emotion Analysis home page!

---

## ✨ Features to Test

1. **Login Page** - Create an account (optional, works without login too)
2. **Speech Test** - Click "Speech Test" and record yourself speaking
   - The app analyzes: fluency, pronunciation, filler words, pace
3. **Interview** - Answer technical questions while being analyzed for:
   - Head pose (looking at camera)
   - Eye contact
   - Attention score
   - Body language
4. **Report** - View detailed analysis with scores and feedback

---

## 🐛 Troubleshooting

### "Python not found"
- Reinstall Python and make sure to check "Add Python to PATH"
- Restart terminal/cmd after installation

### "npm not found"
- Reinstall Node.js from https://nodejs.org/
- Restart terminal/cmd after installation

### "Port 8000 already in use"
- Backend is trying to use port 8000 which is busy
- Kill the process or change port in `backend/main.py`:
  ```python
  port=8001  # Change to 8001 or another number
  ```

### "Port 5173 already in use"
- Frontend is trying to use port 5173 which is busy
- Press `q` in the terminal running the frontend, then try again

### "Camera/Microphone not working"
- Check browser permissions (click lock icon in address bar)
- Allow camera and microphone access
- Make sure no other app is using the camera

### "Backend connection failed"
- Make sure backend is running (`http://localhost:8000` should be accessible)
- Check if you see the "🎉 Backend fully initialized" message
- Restart backend if needed

### "Whisper model download hangs"
- First time setup downloads a large AI model (~1-2 GB)
- Be patient, it might take 5-10 minutes on first run
- You need stable internet connection

### "Out of memory error"
- The speech analyzer needs 4GB+ RAM
- Close other applications
- Restart and try again

---

## 📁 Project Structure

```
Interview_Emotion_Analysis/
├── backend/                    # Python FastAPI server
│   ├── head_pose_detector.py   # Face detection & head pose
│   ├── speech_analyzer.py       # Speech analysis (Whisper AI)
│   ├── server.py               # FastAPI server
│   ├── requirements.txt         # Python dependencies
│   └── main.py                 # Entry point
│
├── frontend/                   # React + TypeScript app
│   ├── src/
│   │   ├── pages/             # Interview, Speech Test, Report pages
│   │   ├── components/        # UI components
│   │   ├── contexts/          # Interview & Auth contexts
│   │   ├── hooks/             # Custom React hooks
│   │   └── lib/               # Firebase, utilities
│   ├── package.json           # Node dependencies
│   └── vite.config.ts         # Frontend build config
│
└── README.md                  # Project documentation
```

---

## 🎓 First Time Usage Tips

1. **For Speech Test:**
   - Speak clearly for at least 20 seconds
   - Choose a quiet room
   - Avoid background noise

2. **For Interview:**
   - Look at the camera throughout the interview
   - Speak naturally but professionally
   - Try to answer all 3 questions

3. **Check Results:**
   - All metrics are scored 0-100
   - Filler words like "um", "uh", "like" are detected
   - You can see your transcription in the report

---

## 🔧 Advanced Setup (Optional)

### Use Custom Whisper Model Size
Edit `backend/main.py`:
```python
# Options: "tiny", "base" (default), "small", "medium", "large"
speech_analyzer = get_speech_analyzer(model_size="small")  # Faster, less accurate
speech_analyzer = get_speech_analyzer(model_size="medium") # Slower, more accurate
```

### Enable Reload on File Changes
Already enabled! Just save files and:
- Backend will auto-reload
- Frontend will hot-reload

---

## 💾 Saving Results

Results can be saved in 2 ways:

1. **Without Login** → Saves locally in browser (lost if cache cleared)
2. **With Firebase Login** → Saves online permanently

To enable Firebase:
1. Set up Firebase project at https://console.firebase.google.com/
2. Get your config values
3. Add them to `frontend/src/lib/firebase.ts`
4. Now login and save feature will work!

---

## 📞 Need Help?

Check the logs in the terminal windows - they usually tell you what's wrong!

Common issues:
- ❌ "Connection refused" → Backend not running
- ❌ "Cannot find module" → Dependencies not installed (run `pip install` or `npm install`)
- ❌ "Timeout" → Whisper model still downloading, wait 5-10 mins

---

## ✅ You're All Set!

Once both servers are running, visit `http://localhost:5173` and start testing! 🎉

Good luck! 🚀
