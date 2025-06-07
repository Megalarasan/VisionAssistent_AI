
# 👁️‍🗨️ Vision AI Assistant 

A **multi-modal AI assistant** that integrates real-time **computer vision**, **speech recognition**, and **Generative AI (Gemini)** to understand, analyze, and respond to voice commands while interpreting the physical world through a camera.

> 🚧 **Project Status:** In Development (Core functionality is working, improvements and UI features in progress)

---

## 🔍 Features

- 🎙️ **Voice-Controlled Interaction**  
  Interact with the assistant through natural voice commands.

- 🧠 **Gemini AI Integration**  
  Uses Google’s Gemini API for:
  - Image understanding (Vision model)
  - Natural language response (Language model)

- 📸 **Real-Time Scene Analysis**  
  Captures images from the webcam and describes the scene using AI.

- 🧾 **Image Memory & Retrieval**  
  Stores and retrieves previous images and lets you query them (e.g., “What is in the last picture?” or “Find the image with a dog and describe it”).

- 🔊 **Text-to-Speech**  
  Converts AI-generated responses into spoken feedback using `pyttsx3`.

---

## 🛠️ Tech Stack

- `Python`
- `OpenCV` – for camera interfacing
- `SpeechRecognition` – for voice commands
- `pyttsx3` – for speech output
- `Google Generative AI (Gemini)` – for vision & language intelligence
- `PIL (Pillow)` – for image handling
- `Regex`, `Threading`, `Datetime`, `NumPy` – utilities & processing

---

## 🧪 Current Capabilities

- Capture and describe scenes on command (e.g., “What is in front of me?”)
- Analyze previous images on request
- Identify objects, people, food, plants, landmarks, etc.
- Answer general questions using Gemini’s language model

---

## 🚀 How to Run

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install opencv-python Pillow numpy SpeechRecognition pyttsx3 google-generativeai
   ```
3. Replace the placeholder API key (`GOOGLE_API_KEY`) in `Main.py` with your actual [Google Generative AI key](https://makersuite.google.com/app/apikey).
4. Run the project:
   ```bash
   python Main.py
   ```

---

## 🧩 Future Improvements

- 🔘 GUI interface for better accessibility
- 🧭 Object detection overlay
- 🌐 Multi-language voice support
- 🗃️ Save image metadata persistently

---

## 📷 Sample Use Cases

- “What object is in front of me?”
- “Is there a cat in the image?”
- “Describe the last image.”
- “Find the image with a tree and tell what else is in it.”

---

## 🤖 Author

Developed by **[Megalarasan S]**  
Pursuing B.Tech in AI and Data Science | Passionate about Future Technologies
