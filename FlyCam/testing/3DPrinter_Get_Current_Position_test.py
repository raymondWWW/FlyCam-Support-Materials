# Test: Can I get current position of the extruder and print it?
# Author: Johnny Duong
# Creation Date: 2-10-2021

# Results: It works (2-10-2021)!
# Serial Says: b'X:50.00Y:150.00Z:50.00E:0.00 Count X: 50.00Y:150.00Z:50.00\nok\n'
# TODO: Convert to string, parse it.

# Load Libraries
import serial
import time
import picamera
import csv
# import numpy
# from PIL import Image
import os


# =============================
# CONSTANTS
# Device Path
# Monoprice Maker Select Pro Ultimate 3D Printer
DEVICE_PATH = '/dev/ttyACM0'
BAUDRATE = 250000
TIMEOUT = 1

# 3D Printer in Lab, uses MicroUSB to USB
# DEVICE_PATH = '/dev/ttyUSB0'
# BAUDRATE = 115200
# TIMEOUT = 1

REBOOT_WAIT_TIME = 5 # In Seconds

# GCode Strings
HOME = "G28"
ABSOLUTE_POS = "G90"
RELATIVE_POS = "G91"
# ==============================

# Hardware Setup
# Connect to 3D printer
printer = serial.Serial(DEVICE_PATH, baudrate = BAUDRATE, timeout=TIMEOUT)

# User Defined Functions

# Function: Initial Setup of 3D Printer
def initial_setup():
	global printer

	# Check if 3D printer is connected
	if printer.isOpen():
		print('Connected to printer')
	 
	 
	# Wait for Printer to Finish Rebooting
	print('Printer is rebooting...\n')
	print("Waiting for", REBOOT_WAIT_TIME, "seconds")
	time.sleep(REBOOT_WAIT_TIME)
	print("Done Waiting, Moving Extruder and Build Plate to Origin/Home\n")
	# printer.write(b'G28\n')
	go_home()
	# printer.write(b'G1Z50\n')
	# printer.write(b'G1X100Y200\n')


# Function: Convert String to binary/Gcode, write to serial
def run_gcode(gcode_string):
	# Converts String to Binary UTF-8 Format for GCode Strings
	#  Adds new line character at the end.
	
	# GCode_string examples:
	#  G28
	#  G1X100Y100
	
	# Call the global printer variable
	global printer
	# Add new line character at the end of the string
	gcode_string = gcode_string + "\n"
	print("Running GCode:", gcode_string)
	
	# Convert to Binary with UTF-8 encoding for string, write to serial
	# printer.write(bytes(gcode_string, "utf-8"))
	printer.write(str.encode(gcode_string))
	# camera.start_preview(fullscreen=False, window=(20, 10, 1366, 768))
	# time.sleep(8)
	# camera.stop_preview()
	output_serial_data()

# Function: Make Extruder and Build Plate Go to Origin/Home.
def go_home():
	# global printer, x_cur, y_cur, z_cur
	# x_cur = 0; y_cur = 0; z_cur = 0;
	run_gcode(HOME)

# Function: Menu, ask for GCODE Input
def menu():
	# Load Menu, will keep looping until user puts in "exit"
	run = True
	while run:
		userInput = input('Type in GCode (or type "exit" to leave): ')
		userInput = userInput.lower()
		
		# print("You entered:", userInput)
		if userInput == "exit":
			print("Exiting...")
			break
		else:
			userInput = userInput.upper()
			run_gcode(userInput)
	pass

# Function: Listen on Serial Port, print results
def output_serial_data():
	global printer
	
	# output = printer.readline()
	output = printer.read(512)
	output = str(output)
	print("Serial Says:", output)
	

# Main
def main():
	# Run Initial Setup
	initial_setup()
	
	# Try GCode to move (proving 3D printer is receiving input)
	menu()
	
	


main()
