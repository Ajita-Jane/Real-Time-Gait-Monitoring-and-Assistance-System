import tkinter as tk
import subprocess
import sys
import os

def run_test():
    """Runs test.py in IDLE."""
    python_exe = sys.executable 
    idle_path = os.path.join(os.path.dirname(python_exe), "Lib", "idlelib", "idle.pyw")
    
    subprocess.run([python_exe, "-m", "idlelib", "-r", "test2.py"], shell=True)


def login():
    """Runs login.py in IDLE."""
    python_exe = sys.executable
    idle_path = os.path.join(os.path.dirname(python_exe), "Lib", "idlelib", "idle.pyw")

    subprocess.run([python_exe, "-m", "idlelib", "-r", "login.py"], shell=True)

root = tk.Tk()
root.title("User Interface")
root.geometry("350x250")
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, padx=20, pady=20, bg="white", relief="ridge", borderwidth=2)
frame.pack(expand=True)

tk.Label(frame, text="Welcome", font=("Arial", 16, "bold"), bg="white").grid(row=0, column=0, columnspan=2, pady=10)

btn_style = {"font": ("Arial", 12), "width": 15, "padx": 5, "pady": 5, "bg": "#4CAF50", "fg": "white"}

register_button = tk.Button(frame, text="Register", command=run_test, **btn_style)
register_button.grid(row=1, column=0, columnspan=2, pady=10)

login_button = tk.Button(frame, text="Login", command=login, **btn_style)
login_button.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()
