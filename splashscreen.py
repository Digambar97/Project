import tkinter as tk
import tkinter.ttk as ttk
import tkinterwidgets as tkw

window = tk.Tk()

window.title("WELCOME PAGE !")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window.geometry("%dx%d+0+0" % (screen_width, screen_height))

window.configure(bg="#AA90F1")

window.resizable(False, False)

bg_image = tk.PhotoImage(file="Resources/Backgrounds/bg1.png")

bg_label = tk.Label(window, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

welcome_label = tkw.Label(window,text="WELCOME TO \n FASHION RECOMMENDER \n SYSTEM AND \n VIRTUAL TRIAL \n ROOM", font=("Comic Sans MS", 32, 'bold'),  fg="#000000", opacity=0.7)
welcome_label.pack(pady=50, anchor="center")

def next_button_click():
    window.destroy()
    import buttons
    next_button.state(["disabled"]) # disable the button to prevent multiple clicks
    style = ttk.Style()
    style.configure("TButton", relief="sunken", background="#c6c6c6")  # configure the style
    window.after(1000, lambda: style.configure("TButton", relief="raised", background="SystemButtonFace"))  # revert the style after 1 second
    window.after(1000, lambda: next_button.state(["!disabled"]))  # enable the button after 1 second


next_button = tk.Button(window, text="NEXT", bg="#bfbfbf", fg="black", font=("Arial", 20), command=next_button_click, relief="raised", borderwidth=2)
next_button.pack(side="right", padx=50)


window.mainloop()