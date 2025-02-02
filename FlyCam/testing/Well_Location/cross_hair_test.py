"""
Cross Hair Test

Algorithm:
GUI, button to Update Image (takes from preview)
GUI, same circle radius as SimpleGUI
Get preview dimensions
Load image
Make a copy of image
Get dimensions of image
Get center x, center y.
On Copy, Draw line, center x, from min y to max y
On Copy, Draw line, center y, from min x to max x
On copy, Draw circle at center x/y with radius
Display Image


TODO:
-Circle overlay opacity? Color change?
-Research ellipse?
-Check if default number, does it change later?
-Color Wheel, get it working. (Use a text or input box to capture the value)
"""
import cv2
import PySimpleGUI as sg

from os.path import join

# Events
LOAD_IMAGE = "Update/Load Image"

# Circle Mask
CIRCLE_RADIUS = 100
CIRCLE_COLOR = (0, 0, 255) # BGR
CIRCLE_THICKNESS = 1

CIRCLE_THICKNESS_KEY = "-=CIRCLE THICKNESS KEY=-"

RAD_KEY = "-RADIUS-"
RAD_MINUS_TEN = "-RAD MINUS_TEN-"
RAD_MINUS_ONE = "-RAD MINUS_ONE-"
RAD_PLUS_ONE = "-RAD PLUS ONE-"
RAD_PLUS_TEN = "-RAD PLUS TEN-"

CIRCLE_EVENT_LIST = [RAD_MINUS_TEN, RAD_MINUS_ONE, RAD_PLUS_ONE, RAD_PLUS_TEN]

LINE_THICKNESS_KEY = "-=LINE THICKNESS KEY=-"
LINE_THICKNESS = 1


ALL_CROSS_HAIR_EVENTS = [LOAD_IMAGE, RAD_MINUS_TEN, RAD_MINUS_ONE, RAD_PLUS_ONE, RAD_PLUS_TEN]


def update_circle(event, values, window, camera):
    global CIRCLE_RADIUS, CIRCLE_THICKNESS

    if event == LOAD_IMAGE:
        CIRCLE_RADIUS = int(values[RAD_KEY])

    if event == RAD_MINUS_TEN:
        CIRCLE_RADIUS -= 10
    elif event == RAD_MINUS_ONE:
        CIRCLE_RADIUS -= 1
    elif event == RAD_PLUS_ONE:
        CIRCLE_RADIUS += 1
    elif event == RAD_PLUS_TEN:
        CIRCLE_RADIUS += 10

    # Update Circle Thickness
    circle_thick_number = int(values[CIRCLE_THICKNESS_KEY])
    CIRCLE_THICKNESS = circle_thick_number

    # Update Window Values for Circle Radius
    window[RAD_KEY].update(CIRCLE_RADIUS)

    # Update Line Thickness
    update_line_thickness(values)

    # Draw/Update the Image
    draw_on_image(camera)


def update_line_thickness(values):
    global LINE_THICKNESS

    # Update LINE_THICKNESS global variable
    thick_number = int(values[LINE_THICKNESS_KEY])
    LINE_THICKNESS = thick_number
    pass


def draw_on_image(camera):

    # Temp image get
    
    #image = get_dummy_image()
    #image_edit = image.copy()
    
    # Get image from camera
    temp_filename = "temp.jpg"
    camera.capture(temp_filename)
    image = cv2.imread(temp_filename)
    image_edit = image.copy()

    # Make a copy of image
    # Get dimensions of image
    # print(image.shape)
    height, width, ch = image.shape
    # Get center x, center y.
    center_x = int(width / 2)
    center_y = int(height / 2)
    # print(f"Center: {center_x, center_y}")

    # On Copy, Draw line, center y, from min x to max x
    # horizontal line
    start_point = (0, center_y)
    end_point = (width, center_y)
    color = (0, 0, 255) # BGR
    image_edit = cv2.line(image_edit, start_point, end_point, color, LINE_THICKNESS)

    # On Copy, Draw line, center x, from min y to max y
    # vertical line
    start_point = (center_x, 0)
    end_point = (center_x, height)
    image_edit = cv2.line(image_edit, start_point, end_point, color, LINE_THICKNESS)

    # On copy, Draw circle at center x/y with radius
    center_coordinates = (center_x, center_y)
    image_edit = cv2.circle(image_edit, center_coordinates, CIRCLE_RADIUS, color, CIRCLE_THICKNESS)

    # Display Image
    cv2.imshow("Cross Hair Preview", image_edit)
    cv2.waitKey(100)
    pass


def get_dummy_image():
    # Dummy Image as placecholder
    
    # Windows
    # image_folder = r'D:\Documents\SF State\Dr. E Lab\Spring 2022\RoboCam\7-25-2022\Code_Pictures_2022-07-25_171105_auto'
    image_file = r'well1_2022-07-25_171110.jpg'
    
    # RPi
    image_folder = r'/home/pi/Projects/3dprinter_sampling/Test Pictures/8-24-2022'
    image_file = r'test_2022-08-24_161919_640x480.jpg'
    
    # image_location
    image_path = join(image_folder, image_file)
    # Dummy resize to 640, 480
    image = cv2.imread(image_path)
    image_resize = cv2.resize(image, (640, 480))

    return image_resize




def get_cross_hair_layout():
    layout = [[sg.Button(LOAD_IMAGE)],
              [sg.Text("Circle Radius:"),
               sg.Button("-10", key=RAD_MINUS_TEN), sg.Button("-1", key=RAD_MINUS_ONE),
               sg.Input(CIRCLE_RADIUS, size=(4, 1), key=RAD_KEY),
               sg.Button("+1", key=RAD_PLUS_ONE), sg.Button("+10", key=RAD_PLUS_TEN)],
              [sg.Text("Circle Thickness:"), sg.Input(CIRCLE_THICKNESS, size=(4, 1), key=CIRCLE_THICKNESS_KEY)],
              [sg.Text("Line Thickness:"), sg.Input(LINE_THICKNESS, size=(4, 1), key=LINE_THICKNESS_KEY)]]
    
    # TODO: Get Color Chooser Button to work
    # [sg.ColorChooserButton("Pick a Color", key="-= COLOR CHOOSER KEY =-")]
    
    return layout


def event_manager(event, values, window, camera):
    
    if event in CIRCLE_EVENT_LIST:
        # print("Circle event detected")
        update_circle(event, values, window, camera)


    if event == LOAD_IMAGE:
        # print(f"Pressed: {LOAD_IMAGE}")

        # Update Line Thickness
        update_line_thickness(values)

        update_circle(event, values, window, camera)

        draw_on_image(camera)
    
    pass
    

def main2():
    print("main2")
    # Sanity Test, does OpenCV actually load the image?
    
    image_folder = r'/home/pi/Projects/3dprinter_sampling/Test Pictures/8-24-2022'
    image_file = r'test_2022-08-24_161919_640x480.jpg'
    image_path = join(image_folder, image_file)
    # Dummy resize to 640, 480
    image = cv2.imread(image_path)
    cv2.imshow("image", image)
    cv2.waitKey(10000)
    pass


def main():
    print("Main")
    
    
    # Setup Camera
    
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32

    # circle_event_list = [RAD_MINUS_TEN, RAD_MINUS_ONE, RAD_PLUS_ONE, RAD_PLUS_TEN]

    # Setup sg theme
    sg.theme("LightGreen")

    # Setup layout
    # Button: Load Image
    # Radius Change, similar to SimpleGUI version.
    """
    layout = [[sg.Button(LOAD_IMAGE)],
              [sg.Text("Circle Radius:"),
               sg.Button("-10", key=RAD_MINUS_TEN), sg.Button("-1", key=RAD_MINUS_ONE),
               sg.Input(CIRCLE_RADIUS, size=(4, 1), key=RAD_KEY),
               sg.Button("+1", key=RAD_PLUS_ONE), sg.Button("+10", key=RAD_PLUS_TEN)],
              [sg.Text("Circle Thickness:"), sg.Input(CIRCLE_THICKNESS, size=(4, 1), key=CIRCLE_THICKNESS_KEY)],
              [sg.Text("Line Thickness:"), sg.Input(LINE_THICKNESS, size=(4, 1), key=LINE_THICKNESS_KEY)],
              [sg.ColorChooserButton("Pick a Color", key="-= COLOR CHOOSER KEY =-")]]
    """
    
    layout = get_cross_hair_layout()
    
    # Setup window
    window = sg.Window("Cross Hair Test", layout=layout)

    # Start while loop for GUI
    while True:
        event, values = window.read()

        # print(event, values)
        
        if event == sg.WIN_CLOSED:
            break
            
        if event in ALL_CROSS_HAIR_EVENTS:
            event_manager(event, values, window, camera)
        """
        if event in CIRCLE_EVENT_LIST:
            # print("Circle event detected")
            update_circle(event, values, window, camera)


        if event == sg.WIN_CLOSED:
            break
        elif event == LOAD_IMAGE:
            # print(f"Pressed: {LOAD_IMAGE}")

            # Update Line Thickness
            update_line_thickness(values)

            update_circle(event, values, window, camera)

            draw_on_image(camera)
        """

    print("Closing GUI...")
    pass


if __name__ == "__main__":
    # Load PiCamera Library
    from picamera import PiCamera
    
    main()
    # main2()


