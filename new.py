import tkinter as tk
import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
from PIL import Image
import tkinterwidgets as tkw
from PIL import ImageTk, ImageDraw

window = tk.Tk()

window.title("Male Choices")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window.geometry("%dx%d+0+0" % (screen_width, screen_height))

bg_image = tk.PhotoImage(file="Resources/Backgrounds/bg4.png")

bg_label = tk.Label(window, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

window.resizable(False, False)

welcome_label = tkw.Label(window, text="Male Fashion", font=("victor mono", 64),  fg="black", opacity=0.7)
welcome_label.pack(side="top", pady=30, padx=2, anchor='w')

def convert_jpg_to_png(jpg_folder, png_folder, min_width=440, min_height=581):
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    for filename in os.listdir(jpg_folder):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(jpg_folder, filename))
            if img.width >= min_width and img.height >= min_height:
                img.save(os.path.join(png_folder, os.path.splitext(filename)[0] + ".png"), "PNG")

def button1_click():
    print("Button 1 clicked")
    window.destroy()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = PoseDetector()

    jpg_folder = "displayed_images/images/images"
    png_folder = "displayed_images/images/png_images"

    # Convert JPG to PNG
    convert_jpg_to_png(jpg_folder, png_folder)

    listShirts = os.listdir(png_folder)

    if not listShirts:
        print("Error: No PNG images found in the png_folder")
        return

    fixedRatio = 262 / 190  
    shirtRatioHeightWidth = 581 / 440
    imageNumber = 0
    imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
    imgButtonLeft = cv2.flip(imgButtonRight, 1)
    counterRight = 0
    counterLeft = 0
    selectionSpeed = 10

    while True:
        success, img = cap.read()
        img = detector.findPose(img, display=False)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

        if lmList:
            path = os.path.join(png_folder, listShirts[imageNumber])
            imgShirt = Image.open(path)

            # Calculate the width of the shirt based on the position of the two points on the user's shoulders
            lm11 = lmList[11]
            lm12 = lmList[12]
            distance = int((lm11[0] - lm12[0]) * fixedRatio)

            # Resize the shirt to fit the user's body
            imgShirt = imgShirt.resize((distance, int(distance * shirtRatioHeightWidth)))

            # Calculate the position of the shirt on the user's body
            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(44 * currentScale), int(48 * currentScale)

            # Overlay the shirt on the user's body
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))

            # Create a button for the shirt
            create_shirt_button(window, imgShirt, lm12[0] - offset[0], lm12[1] - offset[1])

            # Increase or decrease the image number based on the user's position
            if lmList[16][0] < 300:
                counterRight += 1
                if counterRight * selectionSpeed > 360:
                    if imageNumber < len(listShirts) - 1:
                        imageNumber += 1
                        counterRight = 0
            elif lmList[15][0] > 900:
                counterLeft += 1
                if counterLeft * selectionSpeed > 360:
                    if imageNumber > 0:
                        imageNumber -= 1
                        counterLeft = 0

            # Display the updated image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            shirt_image.configure(image=img)
            shirt_image.image = img

        if cv2.waitKey(1) == ord('q'):
            break  # Press 'q' to exit the loop

    cap.release()
    cv2.destroyAllWindows()

def button2_click():
    window.destroy()
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    shirtFolderPath = "displayed_images/images"
    listShirts = os.listdir(shirtFolderPath)

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
        img = cv2.resize(img, (1280, 720))
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
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
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit the loop
            break  # Break the loop if 'q' is pressed

    cap.release()  # Release the video capture
    cv2.destroyAllWindows()  # Close all OpenCV windows

button1 = tk.Button(window, text="T-SHIRTS", bg="#71ffc5", fg="black", height=2, width=16, font=("Comic sans MS", 20), padx=20, pady=10, bd=2, command=button1_click)
button2 = tk.Button(window, text="TROUSERS", bg="#71ffc5", fg="black", height=2, width=16, font=("Comic sans MS", 20), padx=20, pady=10, bd=2, command=button2_click)

button1.pack(padx=45, pady=40, anchor='w')
button2.pack(padx=45, pady=40, anchor='w')

def back_button_click():
    window.destroy()
    import gender

back_button = tk.Button(window, text="BACK", bg="#b3b3ff", fg="black", height=1, font=("Times New Roman", 18), command=back_button_click)
back_button.pack(anchor='se', padx=20, pady=10)

window.mainloop()
