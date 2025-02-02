# PySimpleGUI Radio Testing

import PySimpleGUI as sg

sg.theme("DarkBlue15")

layout = [  [sg.Button("Hello World", size=(20, 4))],
            [sg.Checkbox("Print On:", default=True, key="-IN-")],
            [sg.Radio("Permission Granted", "RADIO1", default=False, key="-IN2-")],
            [sg.Radio("Permission not Granted", "RADIO1", default=True)]]

# Setting Window

window = sg.Window("Push my Buttons", layout, size=(300, 300))


# Show the Application

while True:
    event, values = window.read()
    print(values)
    if event == sg.WIN_CLOSED or event=="Exit":
        break
    elif values["-IN-"] == True and values["-IN2-"] == True:
        print("Hello, World!")


window.close()
