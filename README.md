# 🎬 MovieMind - Movie Recommendation System

A full-stack Movie Recommendation App that suggests similar movies using NLP-based content filtering and displays posters using TMDB API.

---

## 🚀 Live Demo
🔗 https://movie-rec-3eavfxud5emdfatwdcfvky.streamlit.app/

---

## 🧠 How It Works

This project uses a **content-based recommendation system**:

- Combined movie **overview, genres, and tagline**
- Applied **text preprocessing**:
  - Lowercasing
  - Removing punctuation
  - Stopword removal
  - Lemmatization (NLTK)
- Converted text into vectors using **TF-IDF (bi-grams, 50k features)**
- Calculated similarity using **cosine similarity**
- Recommends top-N similar movies

---

## 🔧 Tech Stack

### 💻 Backend
- FastAPI
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK

### 🎨 Frontend
- Streamlit

### 🌐 APIs
- TMDB API (posters, metadata)

### ☁️ Deployment
- Render (Backend)
- Streamlit Cloud (Frontend)

---

## ✨ Features

- 🎥 Browse movies by category (Popular, Top Rated, Trending)
- 🔍 Content-based movie recommendations
- 🖼️ Movie posters from TMDB
- ⚡ Fast and interactive UI
- 🌐 Fully deployed web app

---

## 📁 Project Structure
