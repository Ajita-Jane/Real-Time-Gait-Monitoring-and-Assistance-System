# Real-Time-Gait-Monitoring-and-Assistance-System

An AI-powered desktop application that monitors human gait in real time using webcam input. The system uses MediaPipe Pose Estimation to extract joint-based movement features and classifies gait patterns (normal or abnormal) using an unsupervised autoencoder model. Designed to enhance accessibility, the system enables users with abnormal gait to request support services at public places with automated email notifications to security personnel.

🔍 Features

🎥 Real-Time Gait Detection: Captures webcam video and extracts 47 pose-based features using MediaPipe.

🧠 Unsupervised Gait Classification: Autoencoder model trained to detect abnormal gait patterns with high accuracy.

👤 User Registration System: Secure user onboarding with local data persistence via SQLite.

🆘 On-Demand Service Requests: Users can request assistance (wheelchair, cab, queue skip, etc.), triggering email alerts to security with metadata.

💾 Modular Codebase: Built with scalability in mind to support deployment across multiple service zones.

🛠️ Tech Stack

Computer Vision: OpenCV, MediaPipe

ML Model: Autoencoder

Backend: Python, SQLite

UI: Tkinter

Notifications: SMTP
