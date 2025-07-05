import tkinter as tk
from tkinter import messagebox
import subprocess

def run_enrollment():
    subprocess.run(["python", "enroll.py"])

def run_recognition():
    subprocess.run(["python", "recognize.py"])

def run_database_viewer():
    subprocess.run(["python", "database_viewer.py"])

def exit_app():
    root.quit()

# Create main window
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("500x300")
root.configure(bg="#f0f0f0")

# Title Label
title_label = tk.Label(root, text="Face Recognition System", font=("Arial", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=20)

# Buttons
btn_enroll = tk.Button(root, text="Enroll Face", font=("Arial", 12), command=run_enrollment, width=20)
btn_enroll.pack(pady=10)

btn_recognize = tk.Button(root, text="Recognize Face", font=("Arial", 12), command=run_recognition, width=20)
btn_recognize.pack(pady=10)

btn_database = tk.Button(root, text="Manage Database", font=("Arial", 12), command=run_database_viewer, width=20)
btn_database.pack(pady=10)

btn_exit = tk.Button(root, text="Exit", font=("Arial", 12), command=exit_app, width=20)
btn_exit.pack(pady=10)

# Run main loop
root.mainloop()
