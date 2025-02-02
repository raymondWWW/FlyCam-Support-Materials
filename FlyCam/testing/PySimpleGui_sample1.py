"""
Practice with PySimpleGui

Source:
https://pysimplegui.readthedocs.io/en/latest/#jump-start
"""

import PySimpleGUI as sg

sg.theme("DarkAmber") # Add a touch of color

# ALL the stuff inside your window.
layout = [ [sg.Text("Some text on row 1")],
           [sg.Text("Enter something on Row 2"), sg.InputText()],
           [sg.Button("Ok"), sg.Button("Cancel")] ]

# Create the Window
window = sg.Window("Window Title", layout)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Cancel": #if user closes window or clicks cancel
        break
    print("You entered ", values[0])

window.close()