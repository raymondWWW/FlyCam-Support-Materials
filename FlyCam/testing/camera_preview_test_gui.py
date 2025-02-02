"""
Test Preview Script for PiCamera



Changelog:
16 May 2022: Added in Xlib Psuedo Window Control
"""

from picamera import PiCamera

import cv2
import PySimpleGUI as sg
import random
import time

from Xlib.display import Display

# GUI KEYS
PREVIEW_LOC_X_KEY = "-PREVIEW LOC X KEY-"
PREVIEW_LOC_Y_KEY = "-PREVIEW LOC Y KEY-"
PREVIEW_WIDTH_KEY = "-PREVIEW WIDTH KEY-"
PREVIEW_HEIGHT_KEY = "-PREVIEW HEIGHT KEY-"
ALPHA_KEY = "-ALPHA KEY-"
PREVIEW_KEY_LIST = [PREVIEW_LOC_X_KEY, PREVIEW_LOC_Y_KEY, PREVIEW_WIDTH_KEY, PREVIEW_HEIGHT_KEY, ALPHA_KEY]

# Button Text
START_PREVIEW = "Start Preview"
STOP_PREVIEW = "Stop Preview"

# Camera Preview Settings
PREVIEW_LOC_X = 0
PREVIEW_LOC_Y = 0
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480
PREVIEW_ALPHA = 255
# Opacity, or Alpha (range 0 (invisible) to 255 (opaque))

PREVIOUS_CAMERA_PREVIEW_X = 0
PREVIOUS_CAMERA_PREVIEW_Y = 0

# Displace Pseudo Window to make it easier to grab/see (in pixels?)
PREVIEW_WINDOW_OFFSET = 30

# Xlib Constants
# Default Screen Index, 0 here.
# Assumes one monitor is connected to Raspberry Pi
DEFAULT_SCREEN_INDEX = 0


def get_max_screen_resolution():
    """
    Gets Max Screen Resolution,
    returns max_screen_width, max_screen_height in pixels
    """
    max_screen_width = 0
    max_screen_height = 0
    
    d = Display()
    
    info = d.screen(DEFAULT_SCREEN_INDEX)
    
    max_screen_width = info.width_in_pixels
    max_screen_height = info.height_in_pixels
    
    # print(f"Width: {max_screen_width}, height: {max_screen_height}")
    """
    for screen in range(0,screen_count):
        info = d.screen(screen)
        print("Screen: %s. Default: %s" % (screen, screen==default_screen))
        print("Width: %s, height: %s" % (info.width_in_pixels,info.height_in_pixels))
    """
    
    d.close()
    
    return max_screen_width, max_screen_height


def get_xy_loc_of_all_windows():
    disp = Display()
    root = disp.screen().root
    children = root.query_tree().children
    
    loc_x_list = []
    loc_y_list = []
    
    for win in children:
        winName = win.get_wm_name()
        pid = win.id
        x, y, width, height = get_absolute_geometry(win, root)
        
        loc_x_list.append(x)
        loc_y_list.append(y)
    
    disp.close()
    
    return loc_x_list, loc_y_list


def get_unique_xy_loc():
    loc_x_list, loc_y_list = get_xy_loc_of_all_windows()
    
    # Get unique values from list only, remove negatives
    x_exclude_list = list(set(loc_x_list))
    y_exclude_list = list(set(loc_y_list))
    
    # print("After Set Stuff")
    # print(f"x_exclude_list: {x_exclude_list}")
    # print(f"y_exclude_list: {y_exclude_list}")
    
    # Random Int selection for x and y, exclude unique values above,
    # max would be max screen resolution
    
    # Get max screen width and height
    max_screen_width, max_screen_height = get_max_screen_resolution()
    
    # Use set subtraction to create list of integers for random choice
    # (is faster than using a for loop to remove numbers)
    
    x_start = random.choice(list(set([x for x in range(0, max_screen_width)]) - set(x_exclude_list)))
    y_start = random.choice(list(set([y for y in range(0, max_screen_height)]) - set(y_exclude_list)))
    # print(f"x_start: {x_start}")
    # print(f"y_start: {y_start}")
    
    return x_start, y_start


def get_window_pid(x_start, y_start):
    print("***get_window_pid()***")
    disp = Display()
    root = disp.screen().root
    children = root.query_tree().children
    
    result_pid = 0
    
    for win in children:
        winName = win.get_wm_name()
        pid = win.id
        x, y, width, height = get_absolute_geometry(win, root)
        
        if x == x_start and y == y_start:
            """
            print("======Children=======")
            print(f"winName: {winName}, pid: {pid}")
            print(f"x:{x}, y:{y}, width:{width}, height:{height}")
            """
            # print(f"wm: {win.get_window_title()}")
            
            # Move Window x = 50, y = 20
            # win.configure(x=x+50)
            # win.configure(x=400, y=36)
            
            # Set Window Name to "Camera Preview Window"
            # win.set_wm_name("Camera Preview Window")
            
            result_pid = pid
            break
    
    disp.close()
    
    return result_pid


def get_window_location_from_pid(search_pid):
    # print("get_window_location_from_pid")
    # print(f"search_pid: {search_pid}")
    
    disp = Display()
    root = disp.screen().root
    children = root.query_tree().children
    
    x_win, y_win = 0, 0
    
    for win in children:
        winName = win.get_wm_name()
        pid = win.id
        x, y, width, height = get_absolute_geometry(win, root)
        
        if pid == search_pid:
            """
            print("======Children=======")
            print(f"winName: {winName}, pid: {pid}")
            print(f"x:{x}, y:{y}, width:{width}, height:{height}")
            """
            
            x_win = x
            y_win = y
            
            break
    
    # print(f"x_win:{x_win}, y_win:{y_win}")
    return x_win, y_win
    disp.close()


def move_window_pid(search_pid, x_new, y_new):
    print("***move_window_pid()***")
    # print(f"search_pid: {search_pid}")
    disp = Display()
    root = disp.screen().root
    children = root.query_tree().children
    
    for win in children:
        winName = win.get_wm_name()
        pid = win.id
        x, y, width, height = get_absolute_geometry(win, root)
        
        if pid == search_pid:
            """
            print("======Children=======")
            print(f"winName: {winName}, pid: {pid}")
            print(f"x:{x}, y:{y}, width:{width}, height:{height}")
            """
            
            print(f"Moving Window Name: {winName}, pid: {pid}")
            win.configure(x=x_new, y=y_new)
            
            break
    
    disp.close()


def change_width_height_of_window_pid(search_pid, width_new, height_new):
    print("***change_width_height_of_window_pid()***")
    # print(f"search_pid: {search_pid}")
    disp = Display()
    root = disp.screen().root
    children = root.query_tree().children
    
    for win in children:
        winName = win.get_wm_name()
        pid = win.id
        x, y, width, height = get_absolute_geometry(win, root)
        
        if pid == search_pid:
            """
            print("======Children=======")
            print(f"winName: {winName}, pid: {pid}")
            print(f"x:{x}, y:{y}, width:{width}, height:{height}")
            """
            
            print(f"Change Window Name Width/Height: {winName}, pid: {pid}")
            win.configure(width=width_new, height=height_new)
            
            break
    
    disp.close()


def change_window_name(search_pid, new_window_name):
    print("***change_window_name()***")
    # Change Window Name of Specific PID
    # print(f"search_pid: {search_pid}")
    disp = Display()
    root = disp.screen().root
    children = root.query_tree().children
    
    for win in children:
        winName = win.get_wm_name()
        pid = win.id
        x, y, width, height = get_absolute_geometry(win, root)
        
        if pid == search_pid:
            """
            print("======Children=======")
            print(f"winName: {winName}, pid: {pid}")
            print(f"x:{x}, y:{y}, width:{width}, height:{height}")
            """
            
            win.set_wm_name(new_window_name)
            
            break
    disp.close()


def get_absolute_geometry(win, root):
    """
    Returns the (x, y, height, width) of a window relative to the
    top-left of the screen.
    """
    geom = win.get_geometry()
    (x, y) = (geom.x, geom.y)
    
    # print("Start")
    # print(f"x: {x}, y: {y}")
    
    while True:
        parent = win.query_tree().parent
        pgeom = parent.get_geometry()
        x += pgeom.x
        y += pgeom.y
        
        if parent.id == root.id:
            # print("parent id matches root id. Breaking...")
            break
        win = parent
    
    # print("End")
    # print(f"x: {x}, y: {y}")
    return x, y, geom.width, geom.height


# Define function to check an InputText key for digits only
def check_for_digits_in_key(key_str, window, event, values):
    
    if event == key_str and len(values[key_str]) and values[key_str][-1] not in ('0123456789'):
            # delete last char from input
            # print("Found a letter instead of a number")
            window[key_str].update(values[key_str][:-1])


def main2():
    print("Main2")
    
    global PREVIOUS_CAMERA_PREVIEW_X, PREVIOUS_CAMERA_PREVIEW_Y
    
    # Initialize preview_win_id to store it when GUI is up.
    preview_win_id = 0
    
    # Initialize is_initial_startup flag as True
    is_initial_startup = True
    
    # Preview Window Creation and Tracking
    # Get random/unique x/y window starting position (top-left)
    # loc_x_list, loc_y_list = get_xy_loc_of_all_windows()
    x_start, y_start = get_unique_xy_loc()
    print(f"x_start: {x_start}")
    print(f"y_start: {y_start}")
    
    # Setup PiCamera
    width = 640
    height = 480
    
    camera = PiCamera()
    camera.resolution = (width, height)
    camera.framerate = 32
    
    # Setup PySimpleGUI theme
    sg.theme("LightGreen")
    
    # Setup layout
    layout = [[sg.Text("Preview Location (e.g. x = 0, y = 0):")],
               [sg.Text("x:"), sg.InputText("0", size=(8, 1), enable_events=True, key=PREVIEW_LOC_X_KEY),
                 sg.Text("y:"), sg.InputText("36", size=(8, 1), enable_events=True, key=PREVIEW_LOC_Y_KEY)],
               [sg.Text("Preview Video Size (e.g. width = 640, height = 480):")],
               [sg.Text("width:"), sg.InputText("640", size=(8, 1), enable_events=True, key=PREVIEW_WIDTH_KEY),
                 sg.Text("height:"), sg.InputText("480", size=(8, 1), enable_events=True, key=PREVIEW_HEIGHT_KEY)],
               [sg.Text("Opacity, or Alpha (range 0 (invisible) to 255 (opaque)):"), sg.InputText("255", size=(5, 1), enable_events=True, key=ALPHA_KEY)],
               [sg.Button(START_PREVIEW), sg.Button(STOP_PREVIEW)]
                ]
    layout_p = [[sg.Text("Preview Window. Click and Drag me around to move window!", size=(55, 10))]]
    window_p = sg.Window("Camera Preview Pseudo Window", layout_p, grab_anywhere=True, location=(x_start, y_start))
    
    
    # Setup window
    window = sg.Window("PiCamera Preview Test", layout, location=(500, 50))
    
    # Setup while loop for GUI
    while True:
        event, values = window.read(timeout=1)
        event_p, values_p = window_p.read(timeout=1)
        
        
        # Setup if/else initial_startup condition
        # If initial startup,
        if is_initial_startup == True:
            # print(f"is_initial_startup: {is_initial_startup}")
            # Get PID of Preview Window
            preview_win_id = get_window_pid(x_start, y_start)
            
            # Change Camera Preview Window Name
            new_window_name = "Camera Preview Window"
            change_window_name(preview_win_id, new_window_name)
            # Move This Window to where I want it (0,0)?
            x_new = 400
            y_new = 36
            move_window_pid(preview_win_id, x_new, y_new)
            # Change is_initial_startup to False
            is_initial_startup = False
        else:
            # print(f"is_initial_startup: {is_initial_startup}")
            # get location of Preview Window using PID
            x_win_preview, y_win_preview = get_window_location_from_pid(preview_win_id)
            # print(f"x_win_preview:{x_win_preview}, y_win_preview:{y_win_preview}")
            # camera.start_preview(alpha=255, fullscreen=False, window=(x_win_preview, y_win_preview, 640, 480))
            
            # If previous camera preview x/y is different, update them and call camera.start_preview
            # (Prevents flickering if camera is still)
            # TODO: How to slow down flickering while moving preview window?
            if PREVIOUS_CAMERA_PREVIEW_X != x_win_preview and PREVIOUS_CAMERA_PREVIEW_Y != y_win_preview:
                PREVIOUS_CAMERA_PREVIEW_X = x_win_preview
                PREVIOUS_CAMERA_PREVIEW_Y = y_win_preview
            
                if camera.preview:
                    camera.start_preview(alpha=PREVIEW_ALPHA, fullscreen=False, window=(x_win_preview, y_win_preview + PREVIEW_WINDOW_OFFSET, PREVIEW_WIDTH, PREVIEW_HEIGHT))
            
        
        # Check Input Text for integers only
        
        for preview_key in PREVIEW_KEY_LIST:
            check_for_digits_in_key(preview_key, window, event, values)
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == START_PREVIEW:
            print("Starting Preview With Settings")
            if camera.preview:
                camera.stop_preview()
            prev_width = int(values[PREVIEW_WIDTH_KEY])
            prev_height = int(values[PREVIEW_HEIGHT_KEY])
            prev_loc_x = int(values[PREVIEW_LOC_X_KEY])
            prev_loc_y = int(values[PREVIEW_LOC_Y_KEY])
            alpha_val = int(values[ALPHA_KEY])
            
            # Update Global Variables so Pseudo Window has Control
            PREVIEW_LOC_X = prev_loc_x
            PREVIEW_LOC_Y = prev_loc_y
            PREVIEW_WIDTH = prev_width
            PREVIEW_HEIGHT = prev_height
            PREVIEW_ALPHA = alpha_val
            
            # Move Pseudo Window to input location too
            move_window_pid(preview_win_id, prev_loc_x, prev_loc_y - PREVIEW_WINDOW_OFFSET)
            
            # Change Pseudo Window width and height to match Camera Preview Width/Height + offset
            change_width_height_of_window_pid(preview_win_id, prev_width, prev_height)
            
            camera.start_preview(alpha=alpha_val, fullscreen=False, window=(prev_loc_x, prev_loc_y, prev_width, prev_height))
            
            x_win, y_win = get_window_location_from_pid(preview_win_id)
            print(f"x_win:{x_win}, y_win:{y_win}")
            
        elif event == STOP_PREVIEW:
            print("Stopping Preview")
            camera.stop_preview()
    
    #   If close, break GUI
    #   elif Update Button
    #   elif Start Button
    #   elif Stop Button
    
    window.close()
    camera.close()
    
    
    pass


def main():
    
    width = 640
    height = 480
    
    camera = PiCamera()
    camera.resolution = (width, height)
    camera.framerate = 32
    # rawCapture = PiRGBArray(camera, size=(width, height))
    
    """
    camera.start_preview()
    camera.preview.fullscreen = False
    camera.preview.window = (0, 0, 640, 480)
    """
    
    camera.start_preview(fullscreen=False, window=(0, 0, 640, 480))
    
    time.sleep(5)
    
    camera.stop_preview()
    
    
    pass


if __name__ == "__main__":
    main2()
