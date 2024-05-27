import tkinter as tk
import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
from PIL import ImageTk, Image
import tkinterwidgets as tkw

window = tk.Tk()

window.title("Selection")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window.geometry("%dx%d+0+0" % (screen_width, screen_height))

bg_image = Image.open("Resources/Backgrounds/bg3.png")
bg_image = bg_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(window, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

window.resizable(False, False)

welcome_label = tkw.Label(window, text="PLEASE SELECT YOUR \n GENDER ...  ", font=("Arial", 54 ),  fg="black", opacity=0.7)
welcome_label.pack(side="top", pady=30, anchor="s")

def button1_click():
    window.destroy()
    import male

def button2_click():
    window.destroy()
    import female

button1 = tk.Button(window, text="MALE", bg="#b3ffff", fg="black", height=2, width=20, font=("Comic sans MS", 20), padx=20, pady=10, bd=2, command=button1_click)
button2 = tk.Button(window, text="FEMALE", bg="#ffb3ff", fg="black", height=2, width=20, font=("Comic sans MS", 20), padx=20, pady=10, bd=2, command=button2_click)

button1.pack(padx=20, pady=20)
button2.pack()

window.mainloop()