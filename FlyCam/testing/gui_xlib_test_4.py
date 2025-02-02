"""
Find Window Created by PySimple GUI

Find based on location (top-left)

Code Sources:
https://stackoverflow.com/questions/8705814/get-display-count-and-resolution-for-each-display-in-python-without-xrandr
"""

import PySimpleGUI as sg
import random
import time

from Xlib.display import Display


# Default Screen Index, 0 here.
# Assumes one monitor is connected to Raspberry Pi
DEFAULT_SCREEN_INDEX = 0


# Button Text
START_BUTTON_TEXT = "Start Preview"
STOP_BUTTON_TEXT = "Stop Preview"


# Camera Preview Settings
PREVIEW_LOC_X = 0
PREVIEW_LOC_Y = 0
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480


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


def window_test():
    print("***window_test()***")
    disp = Display()
    root = disp.screen().root
    children = root.query_tree().children
    

    for win in children:
        winName = win.get_wm_name()
        pid = win.id
        x, y, width, height = get_absolute_geometry(win, root)
        
        if winName == None:
        #if x == 300 and y == 20:
            
            print("======Children=======")
            print(f"winName: {winName}, pid: {pid}")
            print(f"x:{x}, y:{y}, width:{width}, height:{height}")
            # print(f"wm: {win.get_window_title()}")
            
            # win.set_wm_name("New Window Name")
            
            # Move Window to the right 50 pixels
            # win.configure(x=x+50)
            
            """
            geom = win.get_geometry()
            # print(f"geom: {geom}")
            print("*****parent*****")
            parent = win.query_tree().parent
            pgeom = parent.get_geometry()
            print(f"pgeom: {pgeom}")
            print(f"parent name: {parent.get_wm_name()}")
            """
            
            print()
            
            """
            if winName == "geany":
                print("======Children=======")
                print(f"winName: {winName}, pid: {pid}")
                get_absolute_geometry(win, root)
                
                geom = win.get_geometry()
                print(f"geom: {geom}")
                print("*****parent*****")
                parent = win.query_tree().parent
                pgeom = parent.get_geometry()
                print(f"pgeom: {pgeom}")
                print(f"parent name: {parent.get_wm_name()}")
                print()
            """
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


def main():
    print("Main")
    
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
    
    # x_start = 300
    # y_start = 20
    
    # Setup SG Theme
    sg.theme("LightGreen")
    
    # Setup Layout
    # TODO: Create buttons for "Start" and "Stop" Preview, to replicate test situation.
    layout = [[sg.Button(START_BUTTON_TEXT, size=(10, 1)), sg.Button(STOP_BUTTON_TEXT, size=(10, 1))]]
    layout_p = [[sg.Text("Preview Window Text", size=(30, 10))]]
    
    # Setup Window
    # TODO: Make sure Control/Window Test has different location than Preview Window
    #   Or add Control location to exclusion list.
    window = sg.Window("Window Test", layout, location=(33, 36))
    window_p = sg.Window("Preview Window", layout_p, grab_anywhere=True, location=(x_start, y_start))
    
    # Start While Loop
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
            
        #   Update Camera Preview locations (variables) to have same location
        #      Check if camera is in preview mode or not
        #      if in preview mode,
        #        Restart camera (if current location is different from old?)
        #      else,
        #        only update preview location variables
        
        
        if event == sg.WIN_CLOSED or event_p == sg.WIN_CLOSED:
            break
        elif event == START_BUTTON_TEXT:
            print(START_BUTTON_TEXT)
            x_win, y_win = get_window_location_from_pid(preview_win_id)
            print(f"x_win:{x_win}, y_win:{y_win}")
            # Start Preview, using location and dimensions
            window_test()
            # preview_win_id = get_window_pid(x_start, y_start)
            # print(f"preview_win_id: {preview_win_id}")
        elif event == STOP_BUTTON_TEXT:
            print(STOP_BUTTON_TEXT)
            # x_new = 400
            # y_new = 36
    
    pass


if __name__ == "__main__":
    main()
