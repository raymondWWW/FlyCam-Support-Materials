"""
Demo program that displays a webcam using OpenCV and applies some very basic image functions

Source:
https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_OpenCV_Simple_GUI.py
"""

import PySimpleGUI as sg
import cv2
import numpy as np

import time

from picamera import PiCamera
from picamera.array import PiRGBArray
import io

from PIL import Image

"""
Demo program that displays a webcam using OpenCV and applies some very basic image functions
- functions from top to bottom -
none:       no processing
threshold:  simple b/w-threshold on the luma channel, slider sets the threshold value
canny:      edge finding with canny, sliders set the two threshold values for the function => edge sensitivity
blur:       simple Gaussian blur, slider sets the sigma, i.e. the amount of blur smear
hue:        moves the image hue values by the amount selected on the slider
enhance:    applies local contrast enhancement on the luma channel to make the image fancier - slider controls fanciness.
"""


def main():
    sg.theme('LightGreen')
    
    # Picamera experiment (works with PiRGBArray)
    """
    width  = 640
    height = 480
    
    camera = PiCamera()
    camera.resolution = (width, height)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    
    time.sleep(0.1)
    
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
    rawCapture.truncate(0)
    """
    
    # 3.2 Capturing to a stream (works!)
    """
    my_file = open("my_image.jpg", "wb")
    camera = PiCamera()
    camera.start_preview()
    time.sleep(2)
    camera.capture(my_file)
    my_file.close()
    """
    
    
    

    # define the window layout
    layout = [
      [sg.Text('OpenCV Demo', size=(60, 1), justification='center')],
      [sg.Image(filename='', key='-IMAGE-')],
      [sg.Radio('None', 'Radio', True, size=(10, 1))],
      [sg.Radio('threshold', 'Radio', size=(10, 1), key='-THRESH-'),
       sg.Slider((0, 255), 128, 1, orientation='h', size=(40, 15), key='-THRESH SLIDER-')],
      [sg.Radio('canny', 'Radio', size=(10, 1), key='-CANNY-'),
       sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='-CANNY SLIDER A-'),
       sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='-CANNY SLIDER B-')],
      [sg.Radio('blur', 'Radio', size=(10, 1), key='-BLUR-'),
       sg.Slider((1, 11), 1, 1, orientation='h', size=(40, 15), key='-BLUR SLIDER-')],
      [sg.Radio('hue', 'Radio', size=(10, 1), key='-HUE-'),
       sg.Slider((0, 225), 0, 1, orientation='h', size=(40, 15), key='-HUE SLIDER-')],
      [sg.Radio('enhance', 'Radio', size=(10, 1), key='-ENHANCE-'),
       sg.Slider((1, 255), 128, 1, orientation='h', size=(40, 15), key='-ENHANCE SLIDER-')],
      [sg.Button('Exit', size=(10, 1))]
    ]

    # create the window and show it without the plot
    window = sg.Window('OpenCV Integration', layout, location=(800, 400))
    
    # OpenCV Prep Work
    
    cap = cv2.VideoCapture(0)
    
    new_width = 640
    new_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    

    while True:
        # Time elapsed range: 0.18 - 0.2 seconds
        start = time.time()
        event, values = window.read(timeout=0)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
      
        # OpenCV Work
        ret, frame = cap.read()
        
        
        # frame = camera.capture(rawCapture, format="rgb", use_video_port=True)
        
        

        if values['-THRESH-']:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
            frame = cv2.threshold(frame, values['-THRESH SLIDER-'], 255, cv2.THRESH_BINARY)[1]
        elif values['-CANNY-']:
            frame = cv2.Canny(frame, values['-CANNY SLIDER A-'], values['-CANNY SLIDER B-'])
        elif values['-BLUR-']:
            frame = cv2.GaussianBlur(frame, (21, 21), values['-BLUR SLIDER-'])
        elif values['-HUE-']:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame[:, :, 0] += int(values['-HUE SLIDER-'])
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        elif values['-ENHANCE-']:
            enh_val = values['-ENHANCE SLIDER-'] / 40
            clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)
        
        end = time.time()
        print(f"Time: {end- start}")

    
    window.close()
    camera.close()


main()
