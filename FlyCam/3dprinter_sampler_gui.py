"""
Graphical User Interface for using the 3D Printer to take picture/video samples
Author: Johnny Duong
Projects: Cell Sensor and MHT
San Francisco State University

Current Features:
-Has Camera Feed
-Can move X, Y, Z of 3D Printer in various relative direction and increments
-Can get Current Location of Extruder Nozzle
-Input Custom GCode

Future Features:
-Smart Movement: Only take a picture or video if current location is
                 the destination (+/- 1 mm)
-Be able to take picture/video without interferring with camera feed
-Save/Open CSV Option for locations
-Preview Sample Locations
-Run Experiment (photo or video, maybe use a radio button)
   -Run for x iterations
-Camera Settings (white balance, sharpness, and so on)
-Display Current Location in GUI

Current TODO List:
-Get Current Location Mananger (runs it twice to get location)
-Put GUI Keys/Text as Constants
-Experiment with Tabs
 Source: https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Tabs_Simple.py
         https://csveda.com/creating-tabbed-interface-using-pysimplegui/

Changelog
25 Apr 2022: Fixed restart bug, can now run multiple experiments without restarting GUI!
             Solution: Use flag to make experiment function end and make forever while loop.
21 Apr 2022: Added in Z Stack Creator
13 Apr 2022: Added Camera Tab to adjust picture capture resolution for "Pic" button and will show resize image.
06 Jun 2021: Can take pictures in Experiment Thread. No video yet. Can't change resolution, bugs out. Buffer issue?
05 Jun 2021: Added in Experiment Thread, can now run GUI and Experiment at the same time.
28 Apr 2021: Changed Experiment variables into CONSTANTS
26 Apr 2021: Added in 2 Tabs: Start Experiment and Movement
18 Apr 2021: Started Changelog, Allow user to input their own GCode.

"""

# Import PySimpleGUI, cv2, numpy, time libraries
# Import picamera libraries

from datetime import datetime
from picamera.array import PiRGBArray, PiBayerArray
from picamera import PiCamera
import csv
import PySimpleGUI as sg
import cv2
import numpy as np
import os
import time
import threading

# Import modules
import settings as C
import get_current_location_m114 as GCL
import printer_connection as printer
import prepare_experiment as P

# ==== USER CONSTANTS - GUI ====
# TODO: Put these in a YAML GUI Settings File?

# ---- EXPERIMENT CONSTANTS ----
OPEN_CSV_PROMPT = "Open CSV:"
OPEN_CSV_FILEBROWSE_KEY = "-CSV_INPUT-"
START_EXPERIMENT = "Start Experiment"
STOP_EXPERIMENT = "Stop Experiment"
MAX_NUMBER_EXPERIMENTAL_RUNS = 1

# ---- RADIO GUI KEYS AND TEXT ----
EXP_RADIO_PIC_KEY = "-RADIO_PIC-"
EXP_RADIO_VID_KEY = "-RADIO_VID-"
EXP_RADIO_PREVIEW_KEY = "-RADIO_PREVIEW-"
EXP_RADIO_GROUP = "RADIO_EXP"
EXP_RADIO_PIC_TEXT = "Picture"
EXP_RADIO_VID_TEXT = "Video"
EXP_RADIO_PREVIEW_TEXT = "Preview"
EXP_RADIO_PROMPT = "For the experiment, choose to take Pictures, Videos, or Preview Only"

# ---- CAMERA TAB ----
# CONSTANTS
PIC_SAVE_FOLDER = r"/home/pi/Projects/3dprinter_sampling"

# Video Streaming:
# Old = 640x480
"""
VID_WIDTH = 640
VID_HEIGHT = 480
"""
VID_WIDTH = 960
VID_HEIGHT = 720
VID_RES = (VID_WIDTH, VID_HEIGHT)

# Image Capture Resolution
# Take a Picture, 12MP: 4056x3040
PIC_WIDTH = 4056
PIC_HEIGHT = 3040
PIC_RES = (PIC_WIDTH, PIC_HEIGHT)

# Monitor Resolution (The one you're using to look at this)
MON_WIDTH = 1920
MON_HEIGHT = 1080
MON_RES = (MON_WIDTH, MON_HEIGHT)

# GUI CONSTANTS
# Button Labels:
UPDATE_CAMERA_TEXT = "Update Camera Settings"

# Camera GUI Keys
CAMERA_ROTATION_KEY = "-ROTATION_INPUT-"
PIC_WIDTH_KEY = "-PIC_WIDTH_INPUT-"
PIC_HEIGHT_KEY = "-PIC_HEIGHT_INPUT-"
PIC_SAVE_FOLDER_KEY = "-PIC_SAVE_FOLDER_INPUT-"


# --- MOVEMENT CONSTANTS ----
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
# TODO: Put in Constants for GCODE Input

# --- Z Stack Constants ----
# INPUT Z STACK PARAMETERS Keys
Z_START_KEY = "-Z_START_KEY-"
Z_END_KEY = "-Z_END_KEY-"
Z_INC_KEY = "-Z_INC_KEY-"

SAVE_FOLDER_KEY = "-SAVE_FOLDER_KEY-"

# Button Text
START_Z_STACK_CREATION_TEXT = "Start Z Stack Creation"


# --- Save a Location Constants ---
SAVE_LOC_BUTTON = "Save Loc Button"

# Create Temp file to store locations into
TEMP_FOLDER = r"/home/pi/Projects/3dprinter_sampling/temp"
TEMP_FILE = r"temp_loc.csv"
TEMP_FULL_PATH = os.path.join(TEMP_FOLDER, TEMP_FILE)

is_running_experiment = False

# ==== USER DEFINED FUNCTIONS =====

# Define function, run_relative(direction, values)
def run_relative(direction, values):
    # Converts input buttons (direction) into GCODE String,
    #  then calls run_gcode from printer module (not implemented in this demo)
    # Inputs: takes string direction (X+, X-, Y+, Y-, Z+, or Z-)
    #         values from window.read()

    # For debugging, uncomment to see if the direction (event) and values are being passed correctly
    # print("direction:", direction)
    # print("values:", values)

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
    printer.run_gcode("G91")
            
    # Run relative_coordinates GCODE created in this function
    printer.run_gcode(relative_coordinates)
#   TODO: Extruder Speed Adjustment


# define get_current_location_manager()
# print("===================================")
# print("You pressed Get Current Location!")
# printer.run_gcode("M114")
# serial_string = printer.get_serial_data()
# if GCL.does_location_exist_m114(serial_string) == True:
    # current_location_dictionary, is_location_found = GCL.parse_m114(serial_string)
    # print(current_location_dictionary)
    # printer.printer.flush()
# else:
    # print("Location Not Found, Try Again")
    # printer.printer.flush()
# TODO: Test out flush, then M114, will this prevent having to do it twice?
#       Update: No, it doesn't help.
# Algorithm:
#  Flush, run M114, set serial data, check, make it run twice
#   if location not found, run again?

# TODO: Include picamera settings

# Thread version
# Define function start_experiment(event, values)
def run_experiment(event, values, thread_event, camera):
    """
    Description: Runs experiment to take a picture, video, or preview (do nothing)
    
    Input: PySimpleGUI window event and values
    """
    # global camera
    print("run_experiment")
    
    # Get CSV Filename
    csv_filename = values[OPEN_CSV_FILEBROWSE_KEY]
    
    # Get Path List from CSV
    path_list = P.get_path_list_csv(csv_filename)
    
    # Get GCODE Location List from path_list
    gcode_string_list = P.convert_list_to_gcode_strings(path_list)
    
    # Go into Absolute Positioning Mode
    printer.run_gcode(C.ABSOLUTE_POS)
    
    # Create New Folder If not in "Preview" Mode
    if values[EXP_RADIO_PREVIEW_KEY] == False:
        folder_path = P.create_and_get_folder_path()
        print("Not in Preview Mode, creating folder:", folder_path)
    
    # Create While loop to check if thread_event is not set (closing)
    count_run = 0
    # while not thread_event.isSet():
    # while True:
    while is_running_experiment:
        
        # TODO: Put in the rest of the code for Pic, Video, Preview from 3dprinter_start_experiment or prepare_experiment
        print("=========================")
        print("Run #", count_run)
        
        well_number = 1
        for location in gcode_string_list:
            # print(gcode_string)
            printer.run_gcode(location)
            print("Going to Well Number:", well_number)
            time.sleep(4)
            if values[EXP_RADIO_PREVIEW_KEY] == True:
                print("Preview Mode is On, only showing preview camera \n")
                # camera.start_preview(fullscreen=False, window=(30, 30, 500, 500))
                # time.sleep(5)
                
                # camera.stop_preview()
            elif values[EXP_RADIO_VID_KEY] == True:
                print("Recording Video Footage")
                file_full_path = P.get_file_full_path(folder_path, well_number)
                # TODO: Change to Video Captures
                # camera.capture(file_full_path)
            elif values[EXP_RADIO_PIC_KEY] == True:
                print("Taking Pictures Only")
                file_full_path = P.get_file_full_path(folder_path, well_number)
                # print(file_full_path)
                
                # Change Image Capture Resolution
                pic_width = PIC_WIDTH
                pic_height = PIC_HEIGHT
            
                #camera.resolution = (pic_width, pic_height)
                
                camera.capture(file_full_path)
                
                # Return to streaming resolution: 640 x 480 (or it will crash)
                # Bug: Crashes anyway because of threading
                #camera.resolution = (VID_WIDTH, VID_HEIGHT)
                # TODO: Look up Camera settings to remove white balance (to deal with increasing brightness)
            # May implement the following to break out of loop first. Helpful for lots of wells
            """    
            if is_running_experiment == False:
                print("Stopping Experiment...")
                return
            """
            well_number += 1
        
        count_run += 1
        
        # Use For Loop to go through each location
        # TODO: Preview doesn't show preview camera
        # Original
        # for location in gcode_string_list:
            # # print(location)
            # printer.run_gcode(location)
            # time.sleep(5)
        
        
    print("=========================")
    print("Experiment Stopped")
    print("=========================")
    
# Takes in event and values to check for radio selection (Pictures, Videos, or Preview)
# Takes in CSV filename or location list generated from opening CSV file
#    Use get_path_list_csv(csv_filename) and convert_list_to_gcode_strings(path_list) from prepare_experiment module
# Create section for camera setup (or create another function to set camera settings)
#  Create function to return camera settings to default (for preview?)
# Create section for video camera setup (length of time to record)
# Goes to each location in list and takes picture, video, or nothing
#   Use for loop to go through each location list
#     Use if statement chain for radio buttons
#       If Picture, take picture. If Video, take video. If Preview, only go there.
# TODO: Include input for number of runs or length of time to run? (Use my Arduino strategy, put in the camera for loop
#       Recommend number of runs first, then implement countdown algorithm?
# TODO: Test picture/video capabilities while camera feed is running. Update, picture works


# Non-thread version
def run_experiment_gui(main_values, camera):
    # Inputs: values or csv_filename?
    
    global is_running_experiment
    
    # Get paths from CSV file
    print("run_experiment")
    
    
    
    
    # Get CSV Filename
    csv_filename = main_values[OPEN_CSV_FILEBROWSE_KEY]
    
    # Get Path List from CSV
    path_list = P.get_path_list_csv(csv_filename)
    
    # Get GCODE Location List from path_list
    gcode_string_list = P.convert_list_to_gcode_strings(path_list)
    gcode_string_list_len = len(gcode_string_list)
    print(f"gcode_string_list_len: {gcode_string_list_len}")
    
    # Go into Absolute Positioning Mode
    printer.run_gcode(C.ABSOLUTE_POS)
    
    # Move to first well
    print("Moving to first well and waiting a few seconds")
    printer.run_gcode(gcode_string_list[0])
    
    # Wait to go to well
    time.sleep(8)
    print("Done waiting to go to WELL")
    
    
    # setup_picture_camera_settings(camera)
    # setup_default_camera_settings(camera)
    
    
    # Change camera resolution
    # Sensor resolution (Pi Camera 2, 3280x2464)
    # Change resolution to largest resolution for taking pictures
    # Change Image Capture Resolution
    pic_width = PIC_WIDTH
    pic_height = PIC_HEIGHT

    camera.resolution = (pic_width, pic_height)
    
    # Sleep time for exposure mode
    # time.sleep(expo_wait_time)
    
    
    # Setup separate GUI
    # setup theme
    sg.theme("Light Brown 3")
    
    # setup layout of new GUI (one window with a single button)
    layout_exp = [[sg.Button("Stop Experiment", size=(20,20))]]

    # setup window for new GUI
    window_exp = sg.Window("Experiment GUI Window", layout_exp, finalize=True)
    
    # Create New Folder If not in "Preview" Mode
    if main_values[EXP_RADIO_PREVIEW_KEY] == False:
        folder_path = P.create_and_get_folder_path()
        print("Not in Preview Mode, creating folder:", folder_path)
    
    # Setup how long to wait before moving to next well (and GUI loop)
    time_to_wait = 2000 # in millisec
    
    # Initialize index for going through gcode_string_list
    index = 0
    # ---- EVENT LOOP ----
    while True:
        event, values = window_exp.read(timeout=time_to_wait)
        
        # Run Experiment
        # print(f"Index: {index}")
        # print(gcode_string_list[index])
        
        well_number = index + 1
        print(f"Well Number: {well_number}")
        
        printer.run_gcode(gcode_string_list[index])
        # Wait to go to well
        time.sleep(7)
        
        if main_values[EXP_RADIO_PREVIEW_KEY] == True:
            print("Preview Mode is On, only showing preview camera \n")
            # camera.start_preview(fullscreen=False, window=(30, 30, 500, 500))
            # time.sleep(5)
            
            # camera.stop_preview()
        elif main_values[EXP_RADIO_VID_KEY] == True:
            print("Recording Video Footage")
            file_full_path = P.get_file_full_path(folder_path, well_number)
            # TODO: Change to Video Captures
            # camera.capture(file_full_path)
        elif main_values[EXP_RADIO_PIC_KEY] == True:
            print("Taking Pictures Only")
            file_full_path = P.get_file_full_path(folder_path, well_number)
            # print(file_full_path)
            
            
            camera.capture(file_full_path)
            # TODO: Look up Camera settings to remove white balance (to deal with increasing brightness)
            time.sleep(2)
            
            
        
        
        # If index is at the end of the list, reset it. else increment it.
        if index == (gcode_string_list_len - 1):
            index = 0
        else:
            index += 1
            
        
        
        if event.startswith("Stop"):
            print("You pressed Stop. Stopping experiment")
            break
    
    window_exp.close()
    
    # Change resolution back to video stream
    camera.resolution = (VID_WIDTH, VID_HEIGHT)
    # time.sleep(expo_wait_time)
    
    # setup_default_camera_settings(camera)
    
    is_running_experiment = False

    
    # 
    pass


# Define function get_gcode_string_list(values)

def get_gcode_string_list(values):
    """
    Description: Takes CSV File from values (GUI Data), returns gcode_string_list
    Input: values, a dictionary from PySimpleGUI Window Reads
    Return/Output: GCode String List for well location.
    """
    # Get CSV Filename
    csv_filename = values[OPEN_CSV_FILEBROWSE_KEY]
    
    # Get Path List from CSV
    path_list = P.get_path_list_csv(csv_filename)
    
    # Get GCODE Location List from path_list
    gcode_string_list = P.convert_list_to_gcode_strings(path_list)
    
    # Return gcode_string_list
    
    pass


# Define function, get_sample(folder_path_sample, values)

def get_sample(folder_path_sample, well_number, values):
    """
    Description: Takes Pic/Vid/Preview Radio Values, then takes a
                 picture, video, or preview (do nothing), stores into
                 folder_path_sample
    Inputs:
      - values, a dictionary from PySimpleGUI Window Reads. The main focus are the Radio values for the Pic/Vid/Preview.
      - folder_path_sample, a string holding the unique folder path for the samples (prevents accidental overwrite)
    Return/Output: Doesn't return anything. TODO: Return True/False if failed or successful?
    """
    
    # Create Unique Filename, call get_file_full_path(folder_path, well_number)
    # Check Experiment Radio Buttons
    #  If Picture is True, take a picture. Save with unique filename
    #  If Video is True, take a video. Save with unique filename
    #  If Preview is True, do nothing or print "Preview Mode"
    
    pass
    
def get_video(camera):
    
    # Create Unique Filename
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d_%H%M%S")
    filename = f"video_{current_time_str}.h264"
    
    # Set Recording Time (in seconds)
    recording_time = int(1 * 60)
    
    camera.start_recording(filename)
    camera.wait_recording(recording_time)
    camera.stop_recording()
    
    print(f"Recorded Video: {filename}")


# Define function to create unique text string using date and time.
def get_unique_id():
    current_time = datetime.now()
    unique_id = current_time.strftime("%Y-%m-%d_%H%M%S")
    # print(f"unique_id: {unique_id}")
    return unique_id


# Define function to check an InputText key for digits only
def check_for_digits_in_key(key_str, window, event, values):
    
    if event == key_str and len(values[key_str]) and values[key_str][-1] not in ('0123456789'):
            # delete last char from input
            # print("Found a letter instead of a number")
            window[key_str].update(values[key_str][:-1])


def create_z_stack(z_start, z_end, z_increment, save_folder_location, camera):
    # Assumes all inputs are floating or integers, no letters!
    print("create_z_stack")
    print("Pausing Video Stream")

    # GCODE Position, goes fastest
    position = "G0"

    # Go into Absolute Mode, "G90"
    # Run GCODE to go into Absolute Mode
    printer.run_gcode(C.ABSOLUTE_POS)

    # Will use absolute location mode to go to each z
    # Alternative, you could use relative and get current location to get z value.
    # Test: Use Get Current Location to compare expected vs actual z.

    # Create Unique folder to save into save_folder_location
    save_folder_path = f"{save_folder_location}/z_stack_{get_unique_id()}"
    
    # Check if folder exists, if not, create it
    if not os.path.isdir(save_folder_path):
        print("Folder Does NOT exist! Making New Folder")
        os.mkdir(save_folder_path)
    else:
        print("Folder Exists")
    
    # print(f"save_folder_path: {save_folder_path}")
    
    # Go to first location, wait x seconds?

    # Mark where we think z_focus is?

    for z in np.arange(z_start, z_end+z_increment, z_increment):
        print(f"z: {z}")
        # Make sure number gets rounded to 2 decimal places (ex: 25.23)

        # Round z to 2 decimal places
        z_rounded = round(z, 2)
        # Fill out with zeroes until 5 characters long, example: 1.2 becomes 01.20
        # For easier viewing purposes depending on OS.
        z_rounded_str = f"{z_rounded}".zfill(5) 

        # Convert z to GCODE
        # GCODE Format: G0Z2.34
        gcode_str = f"{position}Z{z_rounded}"

        print(f"gcode_str: {gcode_str}")

        # Go to z location using printer_connection module's run_gcode
        # Possible bug, could this module be used elsewhere? This code may have to run in the same location as the GUI.
        printer.run_gcode(gcode_str)
        # Wait x seconds for extruder to get to location.
        time.sleep(2)


        # Take Picture and save to folder location
        save_file_name = f"_image_{z_rounded_str}_.jpg"
        save_full_path = f"{save_folder_path}/{save_file_name}"
        
        # Change to max resolution
        camera.resolution = PIC_RES
        
        
        camera.capture(save_full_path)
        
        # Change back to streaming resolution
        camera.resolution = VID_RES

    
    print(f"Done Creating Z Stack at {save_folder_path}")

    pass


# Define function to get current location
def get_current_location():
    printer.run_gcode("M114")
    serial_string = printer.get_serial_data()
    if GCL.does_location_exist_m114(serial_string) == True:
        current_location_dictionary, is_location_found = GCL.parse_m114(serial_string)
        print(current_location_dictionary)
        # printer.printer.flush()
    else:
        print("Location Not Found, Try Again")
    pass
    

def get_current_location2():
    print("Getting Current Location...")
    
    # Init result with negative values for error checking
    # If negative value, then location was not found
    result = {"X": -1.00, "Y": -1.00, "Z": -1.00}
    
    # Number of attempts to check for location (how many times to run for loop)
    num_location_checks = 10
    
    # Number of location grabs until that one is accepted (in case of outdated location)
    num_until_location_accepted = 1
    
    # Init location_found_counter (want loc to be found at least once since old one is stored)
    location_found_counter = 0
    
    num_searches = 0
    for i in range(num_location_checks):
        num_searches += 1
        # Uncomment print statement for debugging
        # print(f"Location Search Attempt: {i}")
        # Run M114 GCODE
        printer.run_gcode("M114")
        serial_string = printer.get_serial_data2()
        if GCL.does_location_exist_m114(serial_string) == True:
            
            current_location_dictionary, is_location_found = GCL.parse_m114(serial_string)
            
            if location_found_counter == 0:
                location_found_counter += 1
                # Uncomment print statement for debugging
                # print("Location Found, but might be outdated. Trying again")
                continue
            elif location_found_counter >= num_until_location_accepted:
                result = current_location_dictionary
                print("Location Found, Stopping Search.")
                break
        else:
            print("Location Not Found, Trying Again...")
            # If location not found, wait a second in case there is a buffer issue?
            # If no data found, get_serial_data2 ran at least 20 times, so used default empty string
            #   Should try again
            """
            print(f"Data Found: {serial_string}")
            if len(serial_string) == 0:
                print("No data found")
            """
            time.sleep(1)
            continue
        
        # Get Serial Data
        # If location exist in serial string, increment location_found_counter by 1, start while loop again
        #   If loc exist and counter is 1, save location
        # If location does not exist, don't increment, start while loop again
    
    print(f"Number of Location Retrieval Attempts: {num_searches}")
    print("**Note: If all coord are -1.00, then location was not found")
    print(f"Location: {result}")
    return result


# Save current location
# Alt: Save to List instead, then have "Save" button?
# Ask user to choose file name and location first?
# Can only save loc if filename is chosen?
def save_current_location():
    print("save_current_location")
    cur_loc_dict = get_current_location2()
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
    print("File Saved")


# define main function
def main():
    
    # Temporary Solution: Make pic res/save globally accessible for modification
    global PIC_WIDTH, PIC_HEIGHT, PIC_SAVE_FOLDER, is_running_experiment

    # Setup Camera
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (VID_WIDTH, VID_HEIGHT)
    camera.framerate = 32
    # MHT: 270
    # camera.rotation = 270

    # Cell Sensor, at home, 90
    # camera.rotation = 90
    
    # MHT: 270, Cell Sensor: 90
    # camera.rotation = C.CAMERA_ROTATION_ANGLE
    # Lab stuff
    camera.rotation = 270
    
    rawCapture = PiRGBArray(camera, size=(VID_WIDTH, VID_HEIGHT))
    
    #
    # allow the camera to warmup
    time.sleep(0.1)
    
    # Setup 3D Printer
    csv_filename = "testing/file2.csv"
    path_list = printer.get_path_list_csv(csv_filename)
    printer.initial_setup(path_list)
    
    # Move Extruder Out Of The Way
    x_start = 0
    y_start = C.Y_MAX
    z_start = 50
    # printer.move_extruder_out_of_the_way(x_start, y_start, z_start)
    
    # Create Temp file to store locations into

    if not os.path.isdir(TEMP_FOLDER):
        os.mkdir(TEMP_FOLDER)
        print(f"Folder does not exist, making directory: {TEMP_FOLDER}")

    # Make newline be blank, prevents extra empty lines from happening
    f = open(TEMP_FULL_PATH, 'w', newline="")
    writer = csv.writer(f)

    # Create headers
    headers = ["X", "Y", "Z"]
    writer.writerow(headers)
    f.close()

    sg.theme("LightGreen")

    # Create tabs layout:
    # Tab 1: Start Experiment (Pic, vid, or Preview), Open CSV File. Disable Start Experiment if no CSV loaded
    # Tab 2: Movement Tab, with input GCODE (temp), Future: Move specific coordinates
    #
    
    # Tab 1: Start Experiment Tab
    # TODO: Create 3 Radio Buttons for Picture, Video, Preview (Default), and Prompt "Choose to take Pictures, Video, or only preview locations"
    # TODO: Create User Input for number of Trials (use placeholder)
    tab_1_layout = [ [sg.Text(OPEN_CSV_PROMPT), sg.Input(), sg.FileBrowse(key=OPEN_CSV_FILEBROWSE_KEY)],
                     [sg.Text(EXP_RADIO_PROMPT)],
                     [sg.Radio(EXP_RADIO_PIC_TEXT, EXP_RADIO_GROUP, default=False, key=EXP_RADIO_PIC_KEY),
                        sg.Radio(EXP_RADIO_VID_TEXT, EXP_RADIO_GROUP, default=False, key=EXP_RADIO_VID_KEY),
                        sg.Radio(EXP_RADIO_PREVIEW_TEXT, EXP_RADIO_GROUP, default=True, key=EXP_RADIO_PREVIEW_KEY)],
                     [sg.Button(START_EXPERIMENT, disabled=True), sg.Button(STOP_EXPERIMENT, disabled=True)]
                   ]
    
    # Tab 2: Movement Tab
    tab_2_layout = [ [sg.Text("", size=(3, 1)), sg.Button("Get Current Location", size=(20, 1)), sg.Button(SAVE_LOC_BUTTON)],
                     [sg.Radio(RELATIVE_TENTH_TEXT, RADIO_GROUP, default=False, key=RELATIVE_TENTH_KEY),
                        sg.Radio(RELATIVE_ONE_TEXT, RADIO_GROUP, default=True, key=RELATIVE_ONE_KEY),
                        sg.Radio(RELATIVE_TEN_TEXT, RADIO_GROUP, default=False, key=RELATIVE_TEN_KEY)
                     ],
                     [sg.Text("", size=(5, 1)), sg.Button(Y_PLUS, size=(10, 1)), sg.Text("", size=(5, 1)), sg.Button(Z_MINUS, size=(5, 1))],
                     [sg.Button(X_MINUS, size=(10, 1)), sg.Button(X_PLUS, size=(10, 1))],
                     [sg.Text("", size=(5, 1)), sg.Button(Y_MINUS, size=(10, 1)), sg.Text("", size=(5, 1)), sg.Button(Z_PLUS, size=(5, 1))],
                     [sg.HorizontalSeparator()],
                     [sg.Text("Input GCODE (e.g. G0X0Y50):")],
                     [sg.InputText(size=(30, 1), key="-GCODE_INPUT-"), sg.Button("Run", size=(5, 1)), sg.Button("Clear", size=(5, 1))]
                   ]
    
    # Setup Tab/GUI Layout
    # Camera Rotation: []
    # Set Still Picture Resolution (Actually changes the constant variables)
    # Width
    # Height
    # Set Camera Settings Button
    tab_3_layout = [ [sg.Text("Camera Rotation (in Degrees):"), sg.InputText("0", size=(10, 1), enable_events=True, key=CAMERA_ROTATION_KEY)],
                     [sg.Text("Set Image Capture Resolution:")],
                     [sg.Text("Pic Width (in pixels):"), sg.InputText(PIC_WIDTH, size=(10, 1), enable_events=True, key=PIC_WIDTH_KEY)],
                     [sg.Text("Pic Height (in pixels):"),sg.InputText(PIC_HEIGHT, size=(10, 1), enable_events=True, key=PIC_HEIGHT_KEY)],
                     [sg.Button(UPDATE_CAMERA_TEXT)],
                     [sg.Text("Save Images to Folder:"), sg.In(size=(25,1), enable_events=True, key=PIC_SAVE_FOLDER_KEY), sg.FolderBrowse()]
                   ]
    
    # Z Stack Tab
    tab_4_layout = [ [sg.Text("Input Z Stack Parameters (Units are in mm):")],
                       [sg.Text("Z Start:"), sg.InputText("0", size=(7, 1), enable_events=True, key=Z_START_KEY),
                        sg.Text("Z End:"),sg.InputText("2", size=(7, 1), enable_events=True, key=Z_END_KEY),
                        sg.Text("Z Inc:"),sg.InputText("0.5", size=(7, 1), enable_events=True, key=Z_INC_KEY)],
                       [sg.Text("Save Folder Location:"), sg.In(size=(25,1), enable_events=True, key=SAVE_FOLDER_KEY), sg.FolderBrowse()],
                       [sg.Button(START_Z_STACK_CREATION_TEXT)]
                   ]
    
    # TABs Layout (New, Experimental
    # TODO: Put in Pic/Video Button, test them out.
    layout = [ [sg.Image(filename='', key='-IMAGE-')],
               [sg.TabGroup([[sg.Tab("Tab 1 (Exp)", tab_1_layout, key="-TAB_1_KEY"),
                              sg.Tab("Tab 2 (Mvmt)", tab_2_layout),
                              sg.Tab("Tab 3 (CAM)", tab_3_layout),
                              sg.Tab("Tab 4 (Z Stack)", tab_4_layout)]])
               ],
               [sg.Button("Pic"), sg.Button("Vid")]
             ]
    
    
    # Define Window Layout (Original)
    # layout = [
        # [sg.Image(filename='', key='-IMAGE-')],
        # [sg.Text("", size=(3, 1)), sg.Button("Get Current Location", size=(20, 1))],
        # [sg.Radio(RELATIVE_TENTH_TEXT, RADIO_GROUP, default=False, key=RELATIVE_TENTH_KEY),
            # sg.Radio(RELATIVE_ONE_TEXT, RADIO_GROUP, default=True, key=RELATIVE_ONE_KEY),
            # sg.Radio(RELATIVE_TEN_TEXT, RADIO_GROUP, default=False, key=RELATIVE_TEN_KEY)],
        # [sg.Text("", size=(5, 1)), sg.Button(Y_PLUS, size=(10, 1)), sg.Text("", size=(5, 1)), sg.Button(Z_MINUS, size=(5, 1))],
        # [sg.Button(X_MINUS, size=(10, 1)), sg.Button(X_PLUS, size=(10, 1))],
        # [sg.Text("", size=(5, 1)), sg.Button(Y_MINUS, size=(10, 1)), sg.Text("", size=(5, 1)), sg.Button(Z_PLUS, size=(5, 1))],
        # [sg.HorizontalSeparator()],
        # [sg.Text("Input GCODE (e.g. G0X0Y50):")],
        # [sg.InputText(size=(30, 1), key="-GCODE_INPUT-"), sg.Button("Run", size=(5, 1)), sg.Button("Clear", size=(5, 1))]
    # ]
    # Have Camera Feed Window
    # To the right, xy, and z
    # Below camera Feed: Show Current Location, Get Current Location Button
    
    # Threading Setup
    # Initialize empty experiment_thread object, will be used with "Start Experiment" is pushed
    experiment_thread = threading.Thread()
    
    # Initialize threading event (Allows you to stop the thread)
    thread_event = threading.Event()
    

    # Create window and show it without plot
    window = sg.Window("3D Printer GUI Test", layout, location=(400, 0))
    
    
    # Create experiment_run_counter
    experiment_run_counter = 0
    # Create Boolean is_running_experiment, default False
    is_running_experiment = False
    # Initialize well_counter to 0 (used for running experiment, going through GCode location list)
    
    # Initialize current_location_dictionary to X=0, Y=0, Z=0
    
    # Initialize folder_path_sample to "" ("Start Experiment" will create unique folder name)
    # **** Note: This for loop may cause problems if the camera feed dies, it will close everything? ****
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # while True:
        event, values = window.read(timeout=0)
        
        # Call Get Current Location Manager Function
        # Print Current Location
        
        # TODO: Create new thread for Current Location display?
        
        # Convert captured frame to array format, then overwrite frame variable (temporary solution)
        frame = frame.array
        
        # If in experiment mode, resize image if it is larger than when rawCapture was created
        # if is_running_experiment == True:
            # Resize frame to size of window, maybe
            # rawCapture = PiRGBArray(camera, size=(VID_WIDTH, VID_HEIGHT))
            # frame = cv2.resize(frame, (VID_WIDTH, VID_HEIGHT))
        
        # TODO: Add in image resizer if in experiment mode. Temp fix to allow for max image resolution while running experiment.
        
        # ---- CSV File Checker and "Start Experiment" Enable/Disable If/Else logic
        # Check if CSV file Exists (length is 0 if CSV not loaded)
        #  Enable "Start Experiment" if true, else disable "Start Experiment"
        if len(values[OPEN_CSV_FILEBROWSE_KEY]) != 0 and is_running_experiment == False:
            # print("CSV File Exists")
            # Enable "Start Experiment" button
            window[START_EXPERIMENT].update(disabled=False)
            # print("values[OPEN_CSV_FILEBROWSE_KEY]:", values[OPEN_CSV_FILEBROWSE_KEY])
            # print(len(values[OPEN_CSV_FILEBROWSE_KEY]))
        else:
            # print("CSV File Does Not Exist")
            # Disable "Start Experiment" button
            window[START_EXPERIMENT].update(disabled=True)
        
        # ---- Main GUI Window If/elif chain ----
        if event == sg.WIN_CLOSED:
            break
        # Tab 1 (Experiment):
        elif event == START_EXPERIMENT:
            print("You pressed Start Experiment")
            
            # Set is_running_experiment to True, we are now running an experiment
            is_running_experiment = True
            
            # Uncomment to see your CSV File (is it the correct path?)
            # print("CSV File:", values[OPEN_CSV_FILEBROWSE_KEY])
            
            # Disable "Start Experiment" Button
            window[START_EXPERIMENT].update(disabled=True)
            # Enable "Stop Experiment" Button
            window[STOP_EXPERIMENT].update(disabled=False)
            
            # Create actual experiment_thread
            experiment_thread = threading.Thread(target=run_experiment, args=(event, values, thread_event, camera,), daemon=True)
            experiment_thread.start()
            
            # Create Unique Folder, Get that Unique Folder's Name
            
            # Non-Thread Version of Running Experiment
            # run_experiment_gui(values, camera)
            
        elif event == STOP_EXPERIMENT:
            print("You pressed Stop Experiment")
            print("Ending experiment after current run")
            experiment_run_counter = 0
            is_running_experiment = False
            # Enable "Start Experiment" Button
            window[START_EXPERIMENT].update(disabled=False)
            # Disable "Stop Experiment" Button
            window[STOP_EXPERIMENT].update(disabled=True)
            
            # Stop thread, set prepares stopping
            thread_event.set()
            
            # Stop experiemnt_thread
            experiment_thread.join(timeout=1)
            
        elif event == "Pic":
            print("You Pushed Pic Button")
            # TODO: Change variables here to Global to match changes in Camera Tab
            # Take a Picture, 12MP: 4056x3040
            pic_width = PIC_WIDTH
            pic_height = PIC_HEIGHT
            unique_id = get_unique_id()
            pic_save_name = f"test_{unique_id}_{pic_width}x{pic_height}.jpg"
            
            camera.resolution = (pic_width, pic_height)
            # camera.resolution = (2592, 1944)
            
            pic_save_full_path = f"{PIC_SAVE_FOLDER}/{pic_save_name}"
            
            camera.capture(pic_save_full_path)
            
            print(f"Saved Image: {pic_save_full_path}")
            
            # Return to streaming resolution: 640 x 480 (or it will crash)
            camera.resolution = (VID_WIDTH, VID_HEIGHT)
            
            """
            # Display image with OpenCV (Keeps Crashing)
            pic_capture = cv2.imread(pic_save_full_path, cv2.IMREAD_COLOR)
            pic_resize = cv2.resize(pic_capture, MON_RES)
            pic_window_tite = "pic_resize"
            cv2.imshow(pic_window_tite, pic_resize)
            print("Press 'q' to close picture")
            key=cv2.waitKey(0)
            if key == ord("q"):
                cv2.destroyAllWindows()
            
            """
            
            #with PiBayerArray(camera) as stream:
                # camera.capture(stream, 'jpeg', bayer=True)
                # Demosaic data and write to output (just use stream.array if you
                # want to skip the demosaic step)
                # output = (stream.demosaic() >> 2).astype(np.uint8)
                #with open('image.data', 'wb') as f:
                    # output.tofile(f)
            
        elif event == "Vid":
            print("You Pushed Vid Button")
            # Take a Video
            get_video(camera)
            
        # Tab 2 (Movement)
        elif event == "Get Current Location":
            print("===================================")
            print("You pressed Get Current Location!")
            get_current_location2()
            """
            printer.run_gcode("M114")
            serial_string = printer.get_serial_data()
            if GCL.does_location_exist_m114(serial_string) == True:
                current_location_dictionary, is_location_found = GCL.parse_m114(serial_string)
                print(current_location_dictionary)
                # printer.printer.flush()
            else:
                print("Location Not Found, Try Again")
                # printer.printer.flush()
            """
        elif event in [X_PLUS, X_MINUS, Y_PLUS, Y_MINUS, Z_PLUS, Z_MINUS]:
            # If any of the direction buttons are pressed, move extruder
            #  in that direction using the increment radio amounts
            run_relative(event, values)
        elif event == "Run":
            # Run GCODE found in the GCode  InputText box
            printer.run_gcode(values["-GCODE_INPUT-"])
        elif event == "Clear":
            # Clear GCode InputText box
            window.FindElement("-GCODE_INPUT-").Update("")
        elif event == UPDATE_CAMERA_TEXT:
            # TAB 3 elif statements
            print("Updating Camera Settings...")
            
            # Update Camera Rotation Angle
            camera_rotation_value = values[CAMERA_ROTATION_KEY]
            camera_rotation_angle = int(camera_rotation_value)
            
            #print(f"Cam Rotation: {camera_rotation_angle}")
            camera.rotation = camera_rotation_angle
            
            # Update Still Image Capture Resolution:
            # global PIC_WIDTH, PIC_HEIGHT
            
            new_pic_width = int(values[PIC_WIDTH_KEY])
            new_pic_height = int(values[PIC_HEIGHT_KEY])
            print(f"New Still Image Resolution: {new_pic_width, new_pic_height}")
            PIC_WIDTH = new_pic_width
            PIC_HEIGHT = new_pic_height
            #print(f"Global: {PIC_WIDTH, PIC_HEIGHT}")
        elif event == START_Z_STACK_CREATION_TEXT:
            print(f"You pressed button: {START_Z_STACK_CREATION_TEXT}")
            z_start = float(values[Z_START_KEY])
            z_end = float(values[Z_END_KEY])
            z_inc = float(values[Z_INC_KEY])
            
            # If nothing chosen, use default folder location:
            if len(values[SAVE_FOLDER_KEY]) == 0:
                save_folder_location = PIC_SAVE_FOLDER
            else:
                save_folder_location = values[SAVE_FOLDER_KEY]
            print(f"save_folder_location: {save_folder_location}")
            create_z_stack(z_start, z_end, z_inc, save_folder_location, camera)
        elif event == SAVE_LOC_BUTTON:
            print(f"You pressed: {SAVE_LOC_BUTTON}")
            save_current_location()
        if event == PIC_SAVE_FOLDER_KEY:
            save_folder = values[PIC_SAVE_FOLDER_KEY]
            print(f"Save folder: {save_folder}")
            PIC_SAVE_FOLDER = save_folder

        
        # print("You entered ", values[0])
        
        # Original
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        
        # Update GUI Window with new image
        window['-IMAGE-'].update(data=imgbytes)
        
        # clear the stream in preparation for the next frame
        # Must do this, else it won't work
        rawCapture.truncate(0)

    # Out of While Loop
    
    # Closing Window
    window.close()
    
    # Closing 3D Printer Serial Connection
    printer.printer.close()
    
    # For loop to show camera feed
    pass

main()
# call main function
