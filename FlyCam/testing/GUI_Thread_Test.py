"""
Sample GUI Test using a thread to run an experiment
"""

# ==== Import Libraries ====
import threading
import time
import PySimpleGUI as sg

import TestThread as T


# ==== USER CONSTANTS ====




# ==== USER DEFINED FUNCTIONS ====

# Define function, run_experiment_thread
def run_experiment_thread(window, e):
    print("run_experiment_thread")
    # event, values = window.read()
    #
    # print(event)
    # print(values)

    dummy_list = [2, 4, 6, 8, 10]
    #
    # max_number_of_experiments = 10
    # experiment_number = 0
    # while experiment_number < max_number_of_experiments:
    #     for item in dummy_list:
    #         time.sleep(1)
    #         print(item)
    #         # if event.startswith("Stop"):
    #         #     print("Stop pressed inside of Run Experiment")

    count = 0
    while not e.isSet():
        count += 1
        print("loop %d" % (count,))
        time.sleep(1.0)
        if count == 10:
            e.set()
            print("Limit reached, stopping?")
    pass



    pass
# Use dummy list, print list

# Define main Function
def main():
    #  Create Layout with a Start and Stop button
    layout = [ [sg.Button("Start"), sg.Button("Stop")] ]
    #  Load Windows
    window = sg.Window("Test Window", layout)

    experiment_thread = threading.Thread()
    test_thread = T.TestThread()
    e = threading.Event()
    #  Run Loop
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event.startswith("Start"):
            print("Start Pushed")
            experiment_thread = threading.Thread(target=run_experiment_thread, args=(window, e,), daemon=True)
            experiment_thread.start()
            # test_thread = T.TestThread()

            # This Works
            # test_thread.start()

        if event.startswith("Stop"):
            print("Stop Pushed")
            e.set()
            experiment_thread.join(timeout=1)

            # This Works
            # test_thread.join()


    #    Start/Stop are opposites: When start is active, Stop is inactive.
    print("Out of While Loop")


main()

