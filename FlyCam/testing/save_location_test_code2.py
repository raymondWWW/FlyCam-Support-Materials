"""
Test Code to Save 3D printer Get Current Location to a CSV File

Test input data is a dictionary containing X/Y/Z float values.
Optional: Have a list of dictionaries to simulate get multiple current locations.

Version 1: Push a button to save to an existing CSV file

Version 2: Display CSV file that allows you to add and delete rows.

Ideas:
-Display "rows" in multiline thing, then save as? (What about random crashes?)
-User can only click on button if file is selected or filename is created?
  -Or when clicking it, it asks user to save file? then will update it?
-Create temp file first, then ask user to save? After saving, delete temp file?
-Create unique temp file at start?

Code Sources:
https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Table_CSV.py
https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Table_Simulation.py
https://www.geeksforgeeks.org/python-save-list-to-csv/

https://www.reddit.com/r/PySimpleGUI/comments/f62gv5/filebrowse_and_filesaveas_buttons_conflicting/
https://stackoverflow.com/questions/61228435/make-a-textbox-in-pysimplegui
https://stackoverflow.com/questions/40193388/how-to-check-if-a-csv-has-a-header-using-python
"""

import csv
import os
import PySimpleGUI as sg


# GUI Constants

SAVE_LOC_BUTTON = "Save Loc Button"

# Create Temp file to store locations into
TEMP_FOLDER = r"D:\Projects\3dprinter_sampling\temp"
TEMP_FILE = r"temp_loc.csv"
TEMP_FULL_PATH = os.path.join(TEMP_FOLDER, TEMP_FILE)

# Dummy Get Current Location, sends a dictionary, simulation M114 parsing function I have.
def get_current_location():
    result = {"X": 1.1, "Y": 2.0, "Z": 3.2}
    # Should always send in x, y, z order
    return result


# Save current location
# Alt: Save to List instead, then have "Save" button?
# Ask user to choose file name and location first?
# Can only save loc if filename is chosen?
def save_current_location():
    print("save_current_location")
    cur_loc_dict = get_current_location()
    print(f"cur_loc_dict: {cur_loc_dict}")

    # Make newline be blank, prevents extra empty lines from happening
    f = open(TEMP_FULL_PATH, 'a', newline="")
    writer = csv.writer(f)

    # Possible to check for headers row?
    # headers = ["X", "Y", "Z"]
    row = []

    for key, value in cur_loc_dict.items():
        print(key, value)
        row.append(value)

    print(row)

    # writer.writerow(headers)
    writer.writerow(row)

    f.close()
    print("File Saved?")


def main():

    # Create Temp file to store locations into
    temp_folder = r"D:\Projects\3dprinter_sampling\temp"
    temp_file = r"temp_loc.csv"

    temp_full_path = os.path.join(temp_folder, temp_file)

    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print(f"Folder does not exist, making directory: {temp_folder}")

    # Make newline be blank, prevents extra empty lines from happening
    f = open(temp_full_path, 'w', newline="")
    writer = csv.writer(f)

    # Create headers
    headers = ["X", "Y", "Z"]
    writer.writerow(headers)
    f.close()


    # Have save_current_location append to temp file
    # Either to multiline to have save as button

    # Set up GUI Layout (A single button)
    layout = [[sg.Button(SAVE_LOC_BUTTON)]]

    # Setup window
    window = sg.Window("Save Location Test", layout, location=(100, 100))

    # Setup text file (hardcoded for now), save as CSV or use CSV library
    # Probably make it a constant, or create on GUI start?

    # Start forever while loop for GUI
    while True:

        event, values = window.read(timeout=1)

        if event == sg.WIN_CLOSED:
            break
        elif event == SAVE_LOC_BUTTON:
            print(f"You pressed: {SAVE_LOC_BUTTON}")
            save_current_location()

    #  Have if statement for closing GUI

    #  If button pressed
    #    Get current location (assume one for now)
    #    Convert dictionary to a CSV row
    #    Append to hardcorded CSV file
    #      Bugs/ideas:
    #       What about duplicates?
    #       Are the rows numbered?
    #       Should I add a text editor or row CSV deleter? What would the user expect for easy editing?
    #       Or is this a "well plate profile"?
    pass


if __name__ == "__main__":
    main()
