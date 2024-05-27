#import streamlit as st
from tkinter import *
from tkinter import  messagebox
from PIL import ImageTk
import pymysql

def clear():
    emailentry.delete(0,END)
    usernameentry.delete(0,END)
    passwordentry.delete(0,END)
    conpasswordentry.delete(0,END)
    check.set(0)

def connect_database():
    if emailentry.get()=='' or usernameentry.get()=='' or passwordentry.get()=='' or conpasswordentry.get()=='':
        messagebox.showerror('Error','All Fields Are Required')
    elif passwordentry.get()!=conpasswordentry.get():
        messagebox.showerror('Error','Password mismatch')
    elif check.get()==0:
        messagebox.showerror('Error', 'Please accept terms and conditions')
    else:
        try:
            conn = pymysql.connect(host='localhost', user='root', password='1234')
            mycursor = conn.cursor()
        except:
            messagebox.showerror('Error','Database connection failed,Please try again')
            return
        try:
            query = 'create database userdata'
            mycursor.execute(query)
            query = 'use userdata'
            mycursor.execute(query)
            query = 'create table data(id int auto_increment primary key not null, email varchar(50), username varchar(100), password varchar(20))'
            mycursor.execute(query)
        except:
            mycursor.execute('use userdata')
        query='select * from data where username=%s'
        mycursor.execute(query,usernameentry.get())

        row=mycursor.fetchone()
        if row!=None:
            messagebox.showerror('Error', 'Username Already Exist')

        else:
            query = 'insert into data(email,username,password) values(%s,%s,%s)'
            mycursor.execute(query, (emailentry.get(), usernameentry.get(), passwordentry.get()))
            conn.commit()
            conn.close()
            messagebox.showinfo('Success', 'Registration is successful')
            clear()
            signup_window.destroy()
            import signin




def login_page():
    signup_window.destroy()
    import signin

signup_window=Tk()
signup_window.title('Signup Page')
signup_window.resizable(False,False)
background=ImageTk.PhotoImage(file='bg.jpg')
bgLabel=Label(signup_window,image=background)
bgLabel.grid()


frame=Frame(signup_window)
frame.place(x=554,y=100)

heading=Label(frame,text='CREATE AN ACCOUNT',font=('Microsoft Yahei UI Light',18,'bold'),bg='white',fg='firebrick1')
heading.grid(row=0,column=0,padx=10,pady=10)

emaillabel=Label(frame,text='Email',font=('Microsoft Yahei UI Light',10,'bold'),bg='white',
                 fg='firebrick1')
emaillabel.grid(row=1,column=0,sticky='w',padx=25)

emailentry=Entry(frame,width=30,font=('Microsoft Yahei UI Light',10,'bold'),fg='white',bg='firebrick1')
emailentry.grid(row=2,column=0,sticky='w',padx=25)


usernamelabel=Label(frame,text='Username',font=('Microsoft Yahei UI Light',10,'bold'),bg='white',
                 fg='firebrick1')
usernamelabel.grid(row=3,column=0,sticky='w',padx=25,pady=(10,0))

usernameentry=Entry(frame,width=30,font=('Microsoft Yahei UI Light',10,'bold'),fg='white',bg='firebrick1')
usernameentry.grid(row=4,column=0,sticky='w',padx=25)

passwordlabel=Label(frame,text='Password',font=('Microsoft Yahei UI Light',10,'bold'),bg='white',
                 fg='firebrick1')
passwordlabel.grid(row=5,column=0,sticky='w',padx=25,pady=(10,0))

passwordentry=Entry(frame,width=30,font=('Microsoft Yahei UI Light',10,'bold'),fg='white',bg='firebrick1')
passwordentry.grid(row=6,column=0,sticky='w',padx=25)

conpasswordlabel=Label(frame,text='Confirm Password',font=('Microsoft Yahei UI Light',10,'bold'),bg='white',
                 fg='firebrick1')
conpasswordlabel.grid(row=7,column=0,sticky='w',padx=25,pady=(10,0))

conpasswordentry=Entry(frame,width=30,font=('Microsoft Yahei UI Light',10,'bold'),fg='white',bg='firebrick1')
conpasswordentry.grid(row=8,column=0,sticky='w',padx=25)

check=IntVar()
termsandconditions=Checkbutton(frame,text='I agree to the terms & conditions',font=('Microsoft Yahei UI Light',9,'bold')
                               ,fg='firebrick1',bg='white',activebackground='white',activeforeground='firebrick1',cursor='hand2',
                               variable=check)
termsandconditions.grid(row=9,column=0,pady=10,padx=15)

signupbutton=Button(frame,text='Signup',font=('Open Sans',16,'bold'),bd=0,bg='firebrick1'
                    ,activebackground='firebrick1',activeforeground='white',width=17,command=connect_database)
signupbutton.grid(row=10,column=0,pady=10)

alreadyaccount=Label(frame,text='Already have an account?',font=('Open Sans',9,'bold'),
                     bg='white',fg='firebrick1')
alreadyaccount.grid(row=11,column=0,sticky='w',padx=25,pady=10)

loginbutton=Button(frame,text='Log in',font=('Open Sans',9,'bold underline'),bg='white',
                   fg='blue',bd=0,cursor='hand2',activeforeground='blue',activebackground='white',
                   command=login_page)
loginbutton.place(x=200,y=395)

signup_window.mainloop()