import sqlite3
import hashlib
import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

result = sys.argv[1] if len(sys.argv) > 1 else "Unknown"

db_path = "users.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        contact TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        gait_result TEXT NOT NULL
    )
''')
conn.commit()
conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def open_ui():
    python_exe = sys.executable
    subprocess.run([python_exe, "-m", "idlelib", "-r", "ui.py"], shell=True)

def register_user():
    name = name_entry.get()
    age = age_entry.get()
    gender = gender_var.get()
    contact = contact_entry.get()
    password = password_entry.get()
    
    if not name or not age or not gender or not contact or not password:
        messagebox.showerror("Input Error", "All fields are required!")
        return

    if not age.isdigit() or int(age) <= 0:
        messagebox.showerror("Input Error", "Please enter a valid age!")
        return
    
    hashed_pw = hash_password(password)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, age, gender, contact, password, gait_result) VALUES (?, ?, ?, ?, ?, ?)", 
                       (name, int(age), gender, contact, hashed_pw, result))
        conn.commit()
        conn.close()
        messagebox.showinfo("Success", "User registered successfully!")
        root.destroy() 
        open_ui() 
    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Contact number already registered!")
    except Exception as e:
        messagebox.showerror("Database Error", str(e))

def toggle_password():
    if password_entry.cget('show') == '*':
        password_entry.config(show='')
        toggle_btn.config(text="Hide")
    else:
        password_entry.config(show='*')
        toggle_btn.config(text="Show")

root = tk.Tk()
root.title("User Registration")
root.geometry("350x450")
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, padx=20, pady=20, bg="white", relief="ridge", borderwidth=2)
frame.pack(expand=True)

tk.Label(frame, text="User Registration", font=("Arial", 16, "bold"), bg="white").grid(row=0, column=0, columnspan=2, pady=10)

tk.Label(frame, text="Gait Analysis Result:", font=("Arial", 12, "bold"), bg="white").grid(row=1, column=0, sticky="w")
gait_label = tk.Label(frame, text=result, font=("Arial", 12), bg="white", fg="#d32f2f" if result == "Abnormal" else "#388e3c")
gait_label.grid(row=1, column=1, pady=5)

tk.Label(frame, text="Name:", font=("Arial", 12), bg="white").grid(row=2, column=0, sticky="w")
name_entry = tk.Entry(frame, font=("Arial", 12), width=22)
name_entry.grid(row=2, column=1, pady=5)

tk.Label(frame, text="Age:", font=("Arial", 12), bg="white").grid(row=3, column=0, sticky="w")
age_entry = tk.Entry(frame, font=("Arial", 12), width=22)
age_entry.grid(row=3, column=1, pady=5)

tk.Label(frame, text="Gender:", font=("Arial", 12), bg="white").grid(row=4, column=0, sticky="w")
gender_var = tk.StringVar(value="Male")
gender_frame = tk.Frame(frame, bg="white")
gender_frame.grid(row=4, column=1, pady=5)
tk.Radiobutton(gender_frame, text="Male", variable=gender_var, value="Male", bg="white").pack(side="left")
tk.Radiobutton(gender_frame, text="Female", variable=gender_var, value="Female", bg="white").pack(side="left")

tk.Label(frame, text="Contact:", font=("Arial", 12), bg="white").grid(row=5, column=0, sticky="w")
contact_entry = tk.Entry(frame, font=("Arial", 12), width=22)
contact_entry.grid(row=5, column=1, pady=5)

tk.Label(frame, text="Password:", font=("Arial", 12), bg="white").grid(row=6, column=0, sticky="w")
password_frame = tk.Frame(frame, bg="white")
password_frame.grid(row=6, column=1, pady=5)
password_entry = tk.Entry(password_frame, font=("Arial", 12), width=18, show="*")
password_entry.pack(side="left")
toggle_btn = tk.Button(password_frame, text="Show", font=("Arial", 10), command=toggle_password)
toggle_btn.pack(side="right")

register_btn = tk.Button(frame, text="Register", font=("Arial", 12, "bold"), width=20, bg="#4CAF50", fg="white", command=register_user)
register_btn.grid(row=7, column=0, columnspan=2, pady=10)

root.mainloop()
