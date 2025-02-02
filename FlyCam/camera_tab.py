"""
Test Code For a Camera Settings Tab for the 3D Printer Sampler GUI

Current Features to Tested:
-[x] Rotate Camera
-[x] Change Image Picture Resolution
-[X] Create Unique ID from date/time
-[x] Choose Save Folder or Create Folder?

Bonus:
-Save Camera Details (or just do that separately)
-Change Video Streaming Video Resolution
-

"""
import cv2
import PySimpleGUI as sg

from datetime import datetime
from picamera import PiCamera
from picamera.array import PiRGBArray, PiBayerArray


# CONSTANTS
PIC_SAVE_FOLDER = ""


# Video Streaming:
VID_WIDTH = 640
VID_HEIGHT = 480

# Image Capture Resolution
PIC_WIDTH = 1920
PIC_HEIGHT = 1080
PIC_RES = (PIC_WIDTH, PIC_HEIGHT)

# GUI CONSTANTS
# Button Labels:
UPDATE_CAMERA_TEXT = "Update Camera Settings"

# Keys
CAMERA_ROTATION_KEY = "-ROTATION_INPUT-"
PIC_WIDTH_KEY = "-PIC_WIDTH_INPUT-"
PIC_HEIGHT_KEY = "-PIC_HEIGHT_INPUT-"
PIC_SAVE_FOLDER_KEY = "-PIC_SAVE_FOLDER_INPUT-"


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

def main():
    print("Main")
    
    global PIC_WIDTH, PIC_HEIGHT, PIC_SAVE_FOLDER
    
    camera = PiCamera()
    camera.resolution = (VID_WIDTH, VID_HEIGHT)
    camera.framerate = 32
    
    rawCapture = PiRGBArray(camera, size=(VID_WIDTH, VID_HEIGHT))
    
    
    # unique_id = get_unique_id()
    # print(f"unique_id: {unique_id}")
    
    # Setup Tab/GUI Layout
    # Camera Rotation: []
    # Set Still Picture Resolution (Actually changes the constant variables)
    # Width
    # Height
    # Set Camera Settings Button
    # Default Value is Zero?
    tab_3_layout = [ [sg.Text("Camera Rotation (in Degrees):"), sg.InputText("0", size=(10, 1), enable_events=True, key=CAMERA_ROTATION_KEY)],
                     [sg.Text("Set Image Capture Resolution:")],
                     [sg.Text("Pic Width (in pixels):"), sg.InputText(PIC_WIDTH, size=(10, 1), enable_events=True, key=PIC_WIDTH_KEY)],
                     [sg.Text("Pic Height (in pixels):"),sg.InputText(PIC_HEIGHT, size=(10, 1), enable_events=True, key=PIC_HEIGHT_KEY)],
                     [sg.Button(UPDATE_CAMERA_TEXT)],
                     [sg.Text("Save Images to Folder:"), sg.In(size=(25,1), enable_events=True, key=PIC_SAVE_FOLDER_KEY), sg.FolderBrowse()]
                   ]
    
    layout = [ [sg.Image(filename='', key='-IMAGE-')],
                tab_3_layout
             ]
    
    
    
    window = sg.Window("Camera Setting Test", layout, location=(800, 400))
    
    # Take Picture Section
    # Choose Folder To Save To
    # Take a Picture button
    
    # Run Camera Loop
    #  Check if Camera Rotation has
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # while True:
        event, values = window.read(timeout=1)
        
        frame = frame.array
        
        # Only allow digits for camera rotation
        
        check_for_digits_in_key(CAMERA_ROTATION_KEY, window, event, values)
        check_for_digits_in_key(PIC_WIDTH_KEY, window, event, values)
        check_for_digits_in_key(PIC_HEIGHT_KEY, window, event, values)
        
        #if event == CAMERA_ROTATION_KEY and len(values[CAMERA_ROTATION_KEY]) and values[CAMERA_ROTATION_KEY][-1] not in ('0123456789'):
            # delete last char from input
            # print("Found a letter instead of a number")
            #window[CAMERA_ROTATION_KEY].update(values[CAMERA_ROTATION_KEY][:-1])
        
        # ---- Main GUI Window If/elif chain ----
        if event == sg.WIN_CLOSED:
            print("Closing GUI")
            break
        elif event == UPDATE_CAMERA_TEXT:
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
        if event == PIC_SAVE_FOLDER_KEY:
            save_folder = values[PIC_SAVE_FOLDER_KEY]
            print(f"Save folder: {save_folder}")
            PIC_SAVE_FOLDER = save_folder
        
        # Original
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        
        # Update GUI Window with new image
        window['-IMAGE-'].update(data=imgbytes)
        
        # clear the stream in preparation for the next frame
        # Must do this, else it won't work
        rawCapture.truncate(0)
    pass


if __name__ == "__main__":
    main()


