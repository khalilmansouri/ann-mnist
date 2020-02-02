import tkFont
from PIL import ImageGrab
import network as NN
import dataLoader as loader
import numpy as np
from PIL import Image, ImageTk
import Tkinter as tk


previous_x = previous_y = 0
x = y = 0
points_recorded = []


def tell_me_where_you_are(event):
    global previous_x, previous_y
    previous_x = event.x
    previous_y = event.y


def draw_from_where_you_are(event):
    global previous_x, previous_y
    global points_recorded
    if points_recorded:
        points_recorded.pop()
        points_recorded.pop()
    x = event.x
    y = event.y
    canvas.create_line(previous_x, previous_y, x, y, fill="white", width=30)
    points_recorded.append(previous_x)
    points_recorded.append(previous_y)
    points_recorded.append(x)
    points_recorded.append(x)
    previous_x = x
    previous_y = y


root = tk.Tk("ANN")
backboard = tk.LabelFrame(root, text="Blackboard")
backboard.pack(fill="both", expand="yes")
canvas = tk.Canvas(backboard, height=280, width=280,
                   bg="black", bd=0, cursor="circle")
canvas.bind("<Motion>", tell_me_where_you_are)
canvas.bind("<B1-Motion>", draw_from_where_you_are)
canvas.pack()
net = NN.load("trainedNet")
resultLabel = tk.StringVar()
probaLabel = tk.StringVar()


def guess(imgData):
    global net
    return net.feedforward(imgData)


def getter(event):
    global root, canvas
    margin = 3
    x = root.winfo_rootx() + canvas.winfo_x() + margin
    y = root.winfo_rooty() + canvas.winfo_y() + margin
    x1 = x + canvas.winfo_width() - margin*2
    y1 = y + canvas.winfo_height() - margin*2
    img = np.array(ImageGrab.grab().crop((x, y, x1, y1)).convert("L"))
    input_size = 280
    output_size = 28
    bin_size = input_size // output_size
    small_image = img.reshape(
        (1, output_size, bin_size, output_size, bin_size)).max(4).max(2)
    data = small_image[0].reshape(-1, 1) / 255.
    ANNOutput = guess(data)
    resultLabel.set(np.argmax(ANNOutput))
    proba = ''
    for i, res in enumerate(ANNOutput):
        proba += str(round(res[0]*100, 1))+"%" + "=" + str(i) + " \n"
    probaLabel.set(proba)


def wipe(event):
    global canvas
    canvas.delete("all")


buttonFrame = tk.Frame(root)
buttonFrame.pack()

button = tk.Button(buttonFrame, text="Guess", fg="red")
button.bind('<Button-1>', getter)
button.pack(side=tk.RIGHT)
button.pack()

clear = tk.Button(buttonFrame, text="Clear", fg="red")
clear.bind('<Button-1>', wipe)
clear.pack(side=tk.LEFT)


probaFrame = tk.LabelFrame(root, text="Results")
probaFrame.pack()
helv100 = tkFont.Font(family="Helvetica", size=100, weight="bold")
label = tk.Label(probaFrame, textvariable=resultLabel,
                 font=helv100, anchor=tk.CENTER)
label.pack(side=tk.RIGHT)

helv30 = tkFont.Font(family="Helvetica", size=10, weight="bold")
proba = tk.Label(probaFrame, textvariable=probaLabel, text="fdsadf",
                 anchor=tk.CENTER, font=helv30)
proba.pack(side=tk.LEFT)
probaLabel.set(
    "0=0%\n1=0%\n2=0%\n3=0%\n4=0%\n5=0%\n0=0%\n6=0%\n7=0%\n8=0%\n9=0%\n")
resultLabel.set("0")
root.mainloop()
