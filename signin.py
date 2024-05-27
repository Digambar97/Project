#import streamlit as st
import tkinter
from tkinter import *
from tkinter import  messagebox
import PIL.ImageTk
import  pymysql

#Func part

def login_user():
    if usernameEntry.get()=='' or PasswordEntry.get()=='':
    # if usernameEntry.get()=='' or password_enter.get()=='':
        messagebox.showerror('Error', 'All Fields Are Required')

    else:
        try:
            conn = pymysql.connect(host='localhost', user='root', password='1234')
            mycursor = conn.cursor()
        except:
            messagebox.showerror('Error','Connection is not established try again')
            return
        query='use userdata'
        mycursor.execute(query)
        query='select * from data where username=%s and password=%s'
        mycursor.execute(query,(usernameEntry.get(),PasswordEntry.get()))
        row=mycursor.fetchone()
        if row==None:
            messagebox.showerror('Error','Invalid username or password')
        else:
            messagebox.showinfo('Congratulations!', 'Login is successful')
            login_window.destroy()
            import splashscreen


def hide():
    openeye.config(file='closeye.png')
    PasswordEntry.config(show='*')
    eyebutton.config(command=show)

def show():
    openeye.config(file='openeye.png')
    PasswordEntry.config(show='')
    eyebutton.config(command=show)



def user_enter(event):
    if usernameEntry.get()=='Username':
        usernameEntry.delete(0,END)

def password_enter(event):
    if PasswordEntry.get()=='Password':
        PasswordEntry.delete(0,END)

def signup_page():
    login_window.destroy()
    import  signup

from PIL import ImageTk
login_window=Tk()
login_window.geometry('990x660+50+50')
login_window.resizable(0,0)
login_window.title('Login Page')
bgimage=ImageTk.PhotoImage(file='bg.jpg')

bglabel=Label(login_window,image=bgimage)
bglabel.place(x=0,y=0)

heading=Label(login_window,text='USER LOGIN',font=('Microsoft Yahei UI Light',23,'bold'),bg='white',fg='firebrick1')
heading.place(x=605,y=120)

usernameEntry=Entry(login_window,width=25,font=('Microsoft Yahei UI Light',11,'bold'),bd=0,fg='firebrick1')
usernameEntry.place(x=580,y=200)
usernameEntry.insert(0,'Username')

usernameEntry.bind('<FocusIn>', user_enter)
frame1=Frame(login_window,width=250,height=2,bg='firebrick1')
frame1.place(x=580,y=222)

PasswordEntry=Entry(login_window,width=25,font=('Microsoft Yahei UI Light',11,'bold'),bd=0,fg='firebrick1')
PasswordEntry.place(x=580,y=260)
PasswordEntry.insert(0,'Password')

PasswordEntry.bind('<FocusIn>', password_enter)
frame2=Frame(login_window,width=250,height=2,bg='firebrick1')
frame2.place(x=580,y=282)
openeye=PhotoImage(file='openeye.png')
eyebutton=Button(login_window,image=openeye,bd=0,bg='white',activebackground='white',cursor='hand2',command=hide)
eyebutton.place(x=800,y=255)


forgetbutton=Button(login_window,text='Forget Password',bd=0,bg='white',activebackground='white',cursor='hand2',
                    font=('Microsoft Yahei UI Light',9,'bold'),fg='firebrick1',activeforeground='firebrick1')
forgetbutton.place(x=715,y=295)

loginButton=Button(login_window,text='Login',font=('Open Sans',16,'bold'),fg='white',bg='firebrick1',activeforeground='white',
                   activebackground='firebrick1',cursor='hand2',bd=0,width=19,command=login_user)

loginButton.place(x=578,y=350)

orLabel=Label(login_window,text='--------------OR--------------',font=('Open Sans',16),fg='firebrick1',bg='white')
orLabel.place(x=583,y=400)

facebook_logo=PhotoImage(file='facebook.png')
fblabel=Label(login_window,image=facebook_logo,bg='white')
fblabel.place(x=640,y=440)


google_logo=PhotoImage(file='google.png')
glabel=Label(login_window,image=google_logo,bg='white')
glabel.place(x=690,y=440)


twitter_logo=PhotoImage(file='twitter.png')
twlabel=Label(login_window,image=twitter_logo,bg='white')
twlabel.place(x=740,y=440)

Signup_Label=Label(login_window,text='Dont have an account?',font=('Open Sans',9,'bold'),fg='firebrick1',bg='white')
Signup_Label.place(x=590,y=500)

newaccButton=Button(login_window,text='Create New One',font=('Open Sans',9,'bold underline'),
                    fg='blue',bg='white',activeforeground='blue',
                   activebackground='firebrick1',cursor='hand2',bd=0,command=signup_page)

newaccButton.place(x=727,y=500)

login_window.mainloop()