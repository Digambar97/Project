import tkinter as tk
import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
from PIL import ImageTk, Image
import tkinterwidgets as tkw

window = tk.Tk()

window.title("Female Choices")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window.geometry("%dx%d+0+0" % (screen_width, screen_height))

bg_image = Image.open("Resources/Backgrounds/bg6.png")
bg_image = bg_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(window, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

window.resizable(False, False)

welcome_label = tkw.Label(window, text="FEMALE FASHION", font=("Arial", 64),  fg="black", opacity=0.7)
welcome_label.pack(side="top", pady=30, anchor="s")

def button1_click():
    window.destroy()

    # cap = cv2.VideoCapture("Resources/Videos/1.mp4")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    detector = PoseDetector()

    shirtFolderPath = "Resources/GirlsDresses"
    listShirts = os.listdir(shirtFolderPath)
    imageNumber=0
    imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
    height, width = imgShirt.shape[:2]
    fixedRatio = width / 500  
    shirtRatioHeightWidth = height / width
    imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
    imgButtonLeft = cv2.flip(imgButtonRight,1)
    counterRight =0
    counterLeft =0
    selectionSpeed = 10

    while True:
        success, img = cap.read()
        # img = cv2.flip(img,1)
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False,draw=False)
  
        if lmList:
            # center = bboxInfo["center"]
            # cv2.circle(img,center,5,(255,0,255),cv2.FILLED)
            lm11 = lmList[11][0:2]
            lm12 = lmList[12][0:2]

            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
            print(widthOfShirt)

            imgShirt = cv2.resize(imgShirt, (widthOfShirt,int(widthOfShirt * shirtRatioHeightWidth)))
            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(14 *currentScale),int(18*currentScale)

            try:
                img = cvzone.overlayPNG(img, imgShirt,(lm12[0]-offset[0],lm12[1]-offset[1]))
            except:
                pass

            img = cvzone.overlayPNG(img, imgButtonRight, (1074, 293))
            img = cvzone.overlayPNG(img, imgButtonLeft, (72, 293))

            if lmList[16][0] < 300 :   #300 is a coordinate 
                counterRight +=1
                cv2.ellipse(img, (139, 360), (66, 66), 0, 0,counterRight * selectionSpeed, (0, 255, 0), 20)
            # in order to generate the eclipse on around the button "ellipse" is used
            # "img"- select the img
            # "(139, 360)" -location (co-ordinates)
            # "(66, 66)" - size
            # "0, 0" - angle and starting angle
            # "counterRight * selectionSpeed"- rotate speed 
            # "(0, 255, 0), 20" -colour and thickness
                if counterRight * selectionSpeed > 360:
                    counterRight = 0
                    if imageNumber < len(listShirts) - 1:
                        imageNumber += 1
            elif lmList[15][0] > 900 :
                counterLeft += 1
                cv2.ellipse(img, (1138, 360), (66, 66), 0, 0,counterLeft * selectionSpeed, (0, 255, 0), 20)
                
                if counterLeft * selectionSpeed > 360:
                    counterLeft = 0
                    if imageNumber > 0:
                        imageNumber -= 1
            else:
                counterRight = 0
                counterLeft = 0

        cv2.imshow("Image", img)
        cv2.waitKey(1)  
   
def button2_click():
    window.destroy()
   # cap = cv2.VideoCapture("Resources/Videos/1.mp4")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    detector = PoseDetector()

    shirtFolderPath = "displayed_images/images/png_images"
    listShirts = os.listdir(shirtFolderPath)
    imageNumber = 0
    # print(listShirts)
    imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
    height, width = imgShirt.shape[:2]
    fixedRatio = width / 600  # widthOfShirt/widthOfPoint11to12
    shirtRatioHeightWidth = height / width
    imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
    imgButtonLeft = cv2.flip(imgButtonRight, 1)
    counterRight = 0
    counterLeft = 0
    selectionSpeed = 10
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        # img = cv2.flip(img,1)
        img = cv2.resize(img, (1280, 720))
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
            # center = bboxInfo["center"]
            lm11 = lmList[11][0:2]
            lm12 = lmList[12][0:2]
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
            # print(widthOfShirt)
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(74 * currentScale), int(78 * currentScale)

            try:
                img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
            except:
                pass

            img = cvzone.overlayPNG(img, imgButtonRight, (1074, 293))
            img = cvzone.overlayPNG(img, imgButtonLeft, (72, 293))

            if lmList[16][0] < 300:
                counterRight += 1
                cv2.ellipse(img, (139, 360), (66, 66), 0, 0, counterRight * selectionSpeed, (0, 255, 0), 20)
                if counterRight * selectionSpeed > 360:
                    counterRight = 0
                    if imageNumber < len(listShirts) - 1:
                        imageNumber += 1
            elif lmList[15][0] > 900:
                counterLeft += 1
                cv2.ellipse(img, (1138, 360), (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)
                if counterLeft * selectionSpeed > 360:
                    counterLeft = 0
                    if imageNumber > 0:
                        imageNumber -= 1

            else:
                counterRight = 0
                counterLeft = 0

        cv2.imshow("Image", img)
        cv2.waitKey(1)

def button3_click():
    window.destroy()
    #cap = cv2.VideoCapture("Resources/Videos/1.mp4")
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    shirtFolderPath = "Resources/GirlsTrousers"
    listShirts = os.listdir(shirtFolderPath)
    # print(listShirts)
    fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
    shirtRatioHeightWidth = 772 / 323
    imageNumber = 0
    imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
    imgButtonLeft = cv2.flip(imgButtonRight, 1)
    counterRight = 0
    counterLeft = 0
    selectionSpeed = 10
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        # img = cv2.flip(img,1)
        img = cv2.resize(img, (1280, 720))
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
            # center = bboxInfo["center"]
            lm11 = lmList[23][1:3]
            lm12 = lmList[24][1:3]
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
            print(widthOfShirt)
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(44 * currentScale), int(48 * currentScale)

            try:
                img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
            except:
                pass

            img = cvzone.overlayPNG(img, imgButtonRight, (1074, 293))
            img = cvzone.overlayPNG(img, imgButtonLeft, (72, 293))

            if lmList[16][1] < 300:
                counterRight += 1
                cv2.ellipse(img, (139, 360), (66, 66), 0, 0,
                            counterRight * selectionSpeed, (0, 255, 0), 20)
                if counterRight * selectionSpeed > 360:
                    counterRight = 0
                    if imageNumber < len(listShirts) - 1:
                        imageNumber += 1
            elif lmList[15][1] > 900:
                counterLeft += 1
                cv2.ellipse(img, (1138, 360), (66, 66), 0, 0,
                            counterLeft * selectionSpeed, (0, 255, 0), 20)
                if counterLeft * selectionSpeed > 360:
                    counterLeft = 0
                    if imageNumber > 0:
                        imageNumber -= 1

            else:
                counterRight = 0
                counterLeft = 0

        cv2.imshow("Image", img)
        cv2.waitKey(1)


button1 = tk.Button(window, text="DRESSES", bg="#ff99e6", fg="black", height=1, width=16, font=("Comic sans MS", 20), padx=20, pady=10, bd=2, command=button1_click)
button2 = tk.Button(window, text="TOPS", bg="#ff99e6", fg="black", height=1, width=16, font=("Comic sans MS", 20), padx=20, pady=10, bd=2, command=button2_click)
# button3 = tk.Button(window, text="TROUSERS", bg="#ff99e6", fg="black", height=1, width=16, font=("Comic sans MS", 20), padx=15, pady=10, bd=2, command=button2_click)

button1.pack(padx=100, pady=20, anchor='s')
button2.pack(padx=100, pady=20, anchor='s')
# button3.pack(padx=100, pady=20, anchor='w')

def back_button_click():
    window.destroy()
    import gender

back_button = tk.Button(window, text="BACK", bg="#b3b3ff", fg="black", height=1, font=("Times New Roman", 18), command=back_button_click)
back_button.pack(anchor='se', padx=20, pady=10)



window.mainloop()