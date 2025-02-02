# PySimpleGUI Radio Testing with 3D Printer Relative Movements
# Without actually using a 3D Printer, only display numbers

# Import Library
import PySimpleGUI as sg

# Create constants for:
# Radio Keys
RELATIVE_TENTH_KEY = "-REL_TENTH-"
RELATIVE_ONE_KEY = "-REL_ONE-"
RELATIVE_TEN_KEY = "-REL_TEN-"
RADIO_GROUP = "RADIO1"
RELATIVE_TENTH_TEXT = "0.10mm"
RELATIVE_ONE_TEXT = "1.00mm"
RELATIVE_TEN_TEXT = "10.00mm"
DEFAULT_DISTANCE = "0.00"

# X+, X-, Y+, Y-, Z+, or Z-
X_PLUS = "X+"
X_MINUS = "X-"
Y_PLUS = "Y+"
Y_MINUS = "Y-"
Z_PLUS = "Z+"
Z_MINUS = "Z-"
# WINDOW_GUI_TIMEOUT
WINDOW_GUI_TIMEOUT = 10 # in ms

# User Defined Functions


# Define function, run_relative(direction, values)
def run_relative(direction, values):
    #   Converts input into GCODE String, then calls run_gcode from printer module (not implemented in this demo)
    #   Inputs: takes string direction (X+, X-, Y+, Y-, Z+, or Z-)
    #           values from window.read()

    # For debugging, uncomment to see if the direction (event) and values are being passed correctly
    # print("direction:", direction)
    # print("values:", values)

    # TODO: Test if Plus (+) character is allowed in GCODE, otherwise use a if/else statement to remove it.

    # Initialize move_amount to 0.00
    move_amount = DEFAULT_DISTANCE

    # Initialize relative_coordinates variable to direction and 0.00 (example: G0X0.00, no movements)
    relative_coordinates = "{}{}".format(direction, move_amount)

    # For debugging, uncomment to see if the formatting matches the example
    # print("relative_coordinates:", relative_coordinates)

    # For debugging, uncomment to see the move_amount before the if/elif chain
    # print("move_amount (before):", move_amount)

    # Use if/elif chain to check which radio button is true (0.1, 1, or 10)
    # If values[-REL_TENTH-] == True
    #  Example If 0.1 true, change relative coordinates to X-0.10
    # else if the values of relative one is True
    #  Make movement amount into 1.00
    # else if the values of relative ten is True
    #  Make movement amount into 1.00
    if values[RELATIVE_TENTH_KEY] == True:
        # print(RELATIVE_TENTH_KEY, "is active")
        # Extract only the float number, ignoring the "mm"
        move_amount = RELATIVE_TENTH_TEXT[0:-2]
    elif values[RELATIVE_ONE_KEY] == True:
        # print(RELATIVE_ONE_KEY, "is active")
        move_amount = RELATIVE_ONE_TEXT[0:-2]
    elif values[RELATIVE_TEN_KEY] == True:
        # print(RELATIVE_TEN_KEY, "is active")
        move_amount = RELATIVE_TEN_TEXT[0:-2]

    # For debugging, uncomment to see the move_amount after the if/elif chain. Did it change?
    # print("move_amount (after):", move_amount)

    #  Use string formatting to create GCode string (example: G0X-1.00)
    relative_coordinates = "G0{}{}".format(direction, move_amount)

    print("relative_coordinates:", relative_coordinates)

    # This is where you would run the GCode
    # Run Relative Mode
    # Run relative_coordinates GCODE created in this function
#   TODO: Extruder Speed Adjustment

# Set PySimpleGUI Theme
sg.theme("DarkAmber")


# Create Layout
# Example:  [] 0.1 mm  [] 1 mm  [] 10 mm
#                       [Up Button]
#               [LEFT]               [Right]
#                         [Down]
#     Keys:    -REL_TENTH-, -REL_ONE-, -REL_TEN-

layout = [ [sg.Radio(RELATIVE_TENTH_TEXT, RADIO_GROUP, default=False, key=RELATIVE_TENTH_KEY),
            sg.Radio(RELATIVE_ONE_TEXT, RADIO_GROUP, default=True, key=RELATIVE_ONE_KEY),
            sg.Radio(RELATIVE_TEN_TEXT, RADIO_GROUP, default=False, key=RELATIVE_TEN_KEY)],
           [sg.Text("", size=(5, 1)), sg.Button(Y_PLUS, size=(10, 1)), sg.Text("", size=(5, 1)), sg.Button(Z_MINUS, size=(5, 1))],
           [sg.Button(X_MINUS, size=(10, 1)), sg.Button(X_PLUS, size=(10, 1))],
           [sg.Text("", size=(5, 1)), sg.Button(Y_MINUS, size=(10, 1)), sg.Text("", size=(5, 1)), sg.Button(Z_PLUS, size=(5, 1))]
]


# Set Window
window = sg.Window("Push my Buttons", layout, size=(300, 300))

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break

    # If event detected is any of the direction buttons, pass event and values to run_relative()
    if event in [X_PLUS, X_MINUS, Y_PLUS, Y_MINUS, Z_PLUS, Z_MINUS]:
        run_relative(event, values)


window.close()

# Use While Loop to Execute GUI
    # get event and values from window.read()
    # Have event to close window

    # Only events/values for if Up Button was pressed
    #    If buttons are pressed, pass on direction (X+, X-, Y+, Y-, Z+, or Z-) and values of radio keys

# Close Window