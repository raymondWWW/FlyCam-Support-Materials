"""
Terminating a Thread Sample Class

Source:
https://www.oreilly.com/library/view/python-cookbook/0596001673/ch06s03.html

Why does run() function run by itself? the Thread object executes the run() function,
so the the run() here is overridding the original Thread object run().

Source:
https://docs.python.org/3/library/threading.html#threading.Thread.run
"""

import threading


class TestThread(threading.Thread):

    def __init__(self, name='TestThread'):
        """ constructor, setting initial variables """
        self._stopevent = threading.Event()
        self._sleepperiod = 1.0

        threading.Thread.__init__(self, name=name)

    def run(self):
        """ main control loop """
        print("%s starts" % (self.getName(),))

        count = 0
        while not self._stopevent.isSet():
            count += 1
            print("loop %d" % (count,))
            self._stopevent.wait(self._sleepperiod)

        print("%s ends" % (self.getName(),))

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        threading.Thread.join(self, timeout)


if __name__ == "__main__":
    testthread = TestThread()
    testthread.start()

    import time

    time.sleep(10.0)

    testthread.join()


