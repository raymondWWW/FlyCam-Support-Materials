"""
Test Code to Save 3D printer Get Current Location to a CSV File

Test input data is a dictionary containing X/Y/Z float values.
Optional: Have a list of dictionaries to simulate get multiple current locations.

Version 1: Push a button to save to an existing CSV file

Version 2: Display CSV file that allows you to add and delete rows.


Code Sources:
https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Table_CSV.py
https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Table_Simulation.py

"""


def main():

    # Set up GUI Layout (A single button)

    # Setup window

    # Setup text file (hardcoded for now), save as CSV or use CSV library
    # Probably make it a constant, or create on GUI start?

    # Start forever while loop for GUI

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
