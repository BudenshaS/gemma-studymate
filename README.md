# 📚 Gemma StudyMate

An AI-powered PDF Study Assistant that helps students learn from their own study materials using **RAG (Retrieval-Augmented Generation)** and **Google Gemma**.

## 🚀 What is Gemma StudyMate?

Gemma StudyMate allows students to upload a subject PDF and interact with it through an AI-powered study assistant.

It can:

- 📄 Read and process PDF study material
- 💬 Answer questions based on the uploaded PDF
- 🧠 Explain topics in simpler language
- 📝 Generate a quick study summary
- 🎯 Create interactive quizzes
- 🔎 Retrieve relevant information from the study material using semantic search

## 🧠 How It Works

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
MiniLM Embeddings
 ↓
FAISS Vector Search
 ↓
Relevant Study Material
 ↓
Gemma
 ↓
Student Answer