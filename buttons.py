import tkinter as tk
import subprocess
import tkinter.ttk as ttk
import tkinterwidgets as tkw
from tkinter.ttk import Label
from PIL import ImageTk, Image

def button1_click():
    window.destroy()
    subprocess.run(["streamlit", "run", "main.py"])

def button2_click():
    window.destroy()
    import gender

window = tk.Tk()
window.title("Our Features")

# Get the width and height of the screen
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set the window size to fullscreen and make it non-resizable
window.geometry("%dx%d+0+0" % (screen_width, screen_height))
window.resizable(False, False)

# Load your image (replace "Resources/Backgrounds/bg6.png" with your actual image file)
bg_image = Image.open("Resources/Backgrounds/bg6.png")
bg_image_resized = bg_image.resize((screen_width, screen_height))
bg_photo = ImageTk.PhotoImage(bg_image_resized)

# Create a label with the image as the background
bg_label = Label(window, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Add other widgets (labels, buttons, etc.)
welcome_label = tkw.Label(window, text="PLEASE SELECT WHAT \n YOU WANT TO TRY \n WITH US...!", font=("Comic Sans MS", 32, 'bold'), fg="#000000", opacity=0.7)
welcome_label.pack(side="top", pady=10, anchor="center")

button1 = tk.Button(window, text="CLOTHES RECOMMENDER", bg="#b3ffb3", fg="black", height=2, width=20, font=("Comic sans MS", 20), padx=20, pady=10, bd=2, command=button1_click, relief="raised", borderwidth=5)
button2 = tk.Button(window, text="VIRTUAL TRIAL ROOM", bg="#ffb3ff", fg="black", height=2, width=20, font=("Comic sans MS", 20), padx=20, pady=10, bd=2, command=button2_click, relief="raised", borderwidth=5)

button1.pack(padx=20, pady=20)
button2.pack()

def back_button_click():
    window.destroy()
    import splashscreen

back_button = tk.Button(window, text="BACK", bg="#bfbfbf", fg="black", height=1, font=("Arial", 18), command=back_button_click)
back_button.pack(side="right", padx=80, pady=10)

window.mainloop()
