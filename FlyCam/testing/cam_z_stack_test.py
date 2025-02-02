"""
Test Code for Z-Stack Capturing using a Raspberry Pi, Camera, and 3D printer.

Test code will only be algorithm that does not include camera and 3D printer.

Algorithm:
-Gets Current Location, assumes sending out of dictionary with location X, Y, and Z (gets z_current)
   - For actual, will have to be called twice to remove old location?
-Runs for loop with z_start, z_end and z_increment
   -z_current will be used in saving image file. Filename format: image_[z].jpg
   -For loop converts z_current float to GCODE string (if I have code that does this, use that)
-Saves images to a unique folder that is created (use my modification of Adrian's code for this)
-GUI:
  -allows user to set z_start, z_end and z_increment
  -Has button to start z_stack creation
  -is a different tab
  -Optional: Displays current location and allows user to input x/y/z for absolute input

Optional:
-Create function to convert float x/y/z to GCODE, and send GCODE.


Code Sources:
https://www.guru99.com/python-range-function.html
https://tutorialdeep.com/knowhow/round-float-to-2-decimal-places-python/

"""

import numpy as np
import os
import PySimpleGUI as sg

from datetime import datetime

# ===== GUI CONSTANTS =====

# INPUT Z STACK PARAMETERS Keys
Z_START_KEY = "-Z_START_KEY-"
Z_END_KEY = "-Z_END_KEY-"
Z_INC_KEY = "-Z_INC_KEY-"

SAVE_FOLDER_KEY = "-SAVE_FOLDER_KEY-"

# Button Text
START_Z_STACK_CREATION_TEXT = "Start Z Stack Creation"


def create_z_stack(z_start, z_end, z_increment, save_folder_location):
    # Assumes all inputs are floating or integers, no letters!
    print("create_z_stack")

    # GCODE Position, goes fastest
    position = "G0"

    # Go into Absolute Mode, "G90"
    # Run GCODE to go into Absolute Mode

    # Will use absolute location mode to go to each z
    # Alternative, you could use relative and get current location to get z value.
    # Test: Use Get Current Location to compare expected vs actual z.

    # Create Unique folder to save into save_folder_location

    # Go to first location, wait x seconds?

    # Mark where we think z_focus is?

    for z in np.arange(z_start, z_end, z_increment):
        print(f"z: {z}")
        # Make sure number gets rounded to 2 decimal places (ex: 25.23)

        # Round z to 2 decimal places
        z_rounded = round(z, 2)

        # Convert z to GCODE
        # GCODE Format: G0Z2.34
        gcode_str = f"{position}Z{z_rounded}"

        print(f"gcode_str: {gcode_str}")

        # Go to z location using printer_connection module's run_gcode
        # Possible bug, could this module be used elsewhere? This code may have to run in the same location as the GUI.


        # Take Picture and save to folder location

        # Wait x seconds to get to location

    # for z in range(z_start, z_end, z_increment):
    #     print(z)

    pass


def create_and_get_folder_path():
    # Get Folder Path
    folder_path = "{}{}".format(C.FOLDERPATH, C.FOLDERNAME_PREFIX)
    # print(folder_path)
    current_time = datetime.now()
    folder_name_suffix = current_time.strftime("%Y-%m-%d_%H%M%S")
    # print(folder_name_suffix)

    folder_path_complete = ""

    # Check if folder exists, if not, create it
    if not os.path.isdir(folder_path_complete):
        print("Folder Does NOT exist! Making New Folder")
        os.mkdir(folder_path_complete)
    else:
        print("Folder Exists")

    # Return Folder Path
    return folder_path_complete


# Define function to create unique text string using date and time.
def get_unique_id():
    current_time = datetime.now()
    unique_id = current_time.strftime("%Y-%m-%d_%H%M%S")
    # print(f"unique_id: {unique_id}")
    return unique_id


# Define function to check an InputText key for digits only
# TODO: Allow floating numbers and a single period only. Periods are allowed at the beginning and end.
def check_for_digits_in_key(key_str, window, event, values):

    if event == key_str and len(values[key_str]) and values[key_str][-1] not in ('0123456789'):
        # delete last char from input
        # print("Found a letter instead of a number")
        window[key_str].update(values[key_str][:-1])


def main():

    # Setup GUI Layout

    # Layout contents:
    #   Text that says unit is in mm
    #   Text boxes for in z_start, z_end, z_increment
    #   Folder Selection of where to save folder
    #   Button to "Create Z Stack"

    layout = [ [sg.Text("Input Z Stack Parameters (Units are in mm):")],
               [sg.Text("Z Start:"), sg.InputText("0", size=(7, 1), enable_events=True, key=Z_START_KEY),
                sg.Text("Z End:"),sg.InputText("2", size=(7, 1), enable_events=True, key=Z_END_KEY),
                sg.Text("Z Inc:"),sg.InputText("0.5", size=(7, 1), enable_events=True, key=Z_INC_KEY)],
               [sg.Text("Save Folder Location:"), sg.In(size=(25,1), enable_events=True, key=SAVE_FOLDER_KEY), sg.FolderBrowse()],
               [sg.Button(START_Z_STACK_CREATION_TEXT)]
               ]

    # Initiliaze GUI Window
    window = sg.Window("Z Stack Test", layout, location=(100, 100))

    # Run forever while loop to start GUI
    while True:
        # Read event and values from window (needed for GUI interaction)
        event, values = window.read(timeout=1)

        # Only allow digits for camera rotation
        # How to allow float numbers or single periods only?


        # ---- Main GUI Window If/elif chain ----
        if event == sg.WIN_CLOSED:
            print("Closing GUI")
            break
        elif event == START_Z_STACK_CREATION_TEXT:
            print(f"You pressed button: {START_Z_STACK_CREATION_TEXT}")
            z_start = float(values[Z_START_KEY])
            z_end = float(values[Z_END_KEY])
            z_inc = float(values[Z_INC_KEY])
            save_folder_location = values[SAVE_FOLDER_KEY]
            print(f"save_folder_location: {save_folder_location}")
            create_z_stack(z_start, z_end, z_inc, save_folder_location)
            # create_z_stack(1.1, 5.2, 0.2)
    pass


if __name__ == "__main__":
    main()
