import sqlite3
import hashlib
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import subprocess
import json
import tempfile
import os
import sys

db_path = "users.db"
excel_path = r"D:\Jane\1. College\8. Semester VIII\Auto Encoders\Location.xlsx"
temp_json_path = os.path.join(tempfile.gettempdir(), "services.json")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user():
    username = username_entry.get()
    password = password_entry.get()
    
    if not username or not password:
        messagebox.showerror("Input Error", "Please enter username and password!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, password FROM users WHERE name = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and user[1] == hash_password(password):
        show_location_services(username)
    else:
        messagebox.showerror("Login Failed", "Invalid username or password!")

def show_location_services(username):
    for widget in frame.winfo_children():
        widget.destroy()

    root.geometry("400x600")
    tk.Label(frame, text=f"Welcome, {username}!", font=("Arial", 14, "bold"), bg="white").grid(row=0, column=0, columnspan=2, pady=10)

    tk.Label(frame, text="Enter Your Location:", font=("Arial", 12), bg="white").grid(row=1, column=0, pady=5, sticky="w")
    location_entry = tk.Entry(frame, font=("Arial", 12), width=25)
    location_entry.grid(row=1, column=1, pady=5)

    tk.Label(frame, text="Select Assistance:", font=("Arial", 12, "bold"), bg="white").grid(row=2, column=0, columnspan=2, pady=10)

    global wheelchair_var, queue_var, ride_var
    wheelchair_var = tk.IntVar()
    queue_var = tk.IntVar()
    ride_var = tk.IntVar()

    tk.Checkbutton(frame, text="Wheelchair Support", variable=wheelchair_var, font=("Arial", 12), bg="white").grid(row=3, column=0, columnspan=2, sticky="w")
    tk.Checkbutton(frame, text="Queue Avoidance", variable=queue_var, font=("Arial", 12), bg="white").grid(row=4, column=0, columnspan=2, sticky="w")
    tk.Checkbutton(frame, text="Ride Booking", variable=ride_var, font=("Arial", 12), bg="white").grid(row=5, column=0, columnspan=2, sticky="w")

    global user_info_label
    user_info_label = tk.Label(frame, text="", font=("Arial", 12, "bold"), fg="blue", bg="white")
    user_info_label.grid(row=6, column=0, columnspan=2, pady=10)

    tk.Button(frame, text="Request Services", font=("Arial", 12), bg="#4CAF50", fg="white",
              command=lambda: request_services(location_entry.get(), username)).grid(row=7, column=0, columnspan=2, pady=10)

def request_services(location, username):
    if not location:
        messagebox.showerror("Input Error", "Please enter your location!")
        return

    selected_services = []
    if wheelchair_var.get():
        selected_services.append("Wheelchair Support")
    if queue_var.get():
        selected_services.append("Queue Avoidance")
    if ride_var.get():
        selected_services.append("Ride Booking")

    if not selected_services:
        messagebox.showerror("Selection Error", "Please select at least one service!")
        return

    try:
        df = pd.read_excel(excel_path)
        df_filtered = df[df["Location"].str.lower() == location.lower()]

        if df_filtered.empty:
            user_info_label.config(text=f"No users found in {location}.")
            return

        result_message = "Services Requested:\n" + ", ".join(selected_services) + "\n\nPlease Contact:\n"
        for _, row in df_filtered.iterrows():
            result_message += f"Name: {row['Name']}, Phone: {row['Phone Number']}\n"

        user_info_label.config(text=result_message, fg="black")

        data_to_pass = {
            "username": username,
            "location": location,
            "services": selected_services
        }
        with open(temp_json_path, "w") as f:
            json.dump(data_to_pass, f)

        python_exe = sys.executable
        subprocess.run([python_exe, "-m", "idlelib", "-r", "mail.py"], shell=True)

    except Exception as e:
        user_info_label.config(text=f"Error: {str(e)}", fg="red")

root = tk.Tk()
root.title("User Login")
root.geometry("400x350")
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, padx=20, pady=20, bg="white", relief="ridge", borderwidth=2)
frame.pack(expand=True, fill="both")

tk.Label(frame, text="Login", font=("Arial", 16, "bold"), bg="white").grid(row=0, column=0, columnspan=2, pady=10)

tk.Label(frame, text="Username:", font=("Arial", 12), bg="white").grid(row=1, column=0, pady=5, sticky="w")
username_entry = tk.Entry(frame, font=("Arial", 12), width=25)
username_entry.grid(row=1, column=1, pady=5)

tk.Label(frame, text="Password:", font=("Arial", 12), bg="white").grid(row=2, column=0, pady=5, sticky="w")
password_entry = tk.Entry(frame, font=("Arial", 12), width=25, show="*")
password_entry.grid(row=2, column=1, pady=5)

login_btn = tk.Button(frame, text="Login", command=authenticate_user, font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5)
login_btn.grid(row=3, column=0, columnspan=2, pady=15)

root.mainloop()
