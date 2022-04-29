
from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

class Direct_fire_timer:
    """
    This class is made for calling certain procedure when the time is reached by timer or when need to call directly
    """
    def start(self, seconds, proc):
        self.proc = proc
        self.timer = Timer(seconds, proc)
        self.timer.start()
    def cancel(self):
        self.timer.cancel()
        self.timer = None
    def direct_proc(self):
        self.cancel()
        self.proc()

def convert_seconds(seconds):
    """
    Changing seconds to h, m, s

    :param seconds: seconds(int)

    return: hour, minute, second
    """
    h = int(seconds / 3600)
    m = int((seconds - 3600 * h) / 60)
    s = (seconds - 3600 * h) % 60
    
    return h, m, s

def convert_time(h, m, s):
    """
    Changing h, m, s to seconds

    :param h: hour(int)
    :param m: minute(int)
    :param s: second(int)

    return: second
    """
    return h * 3600 + m * 60 + s

if __name__ == "__main__":
    def hello(name):
        print("Hello", name)

    def helloworld():
        print("hello world!")

    rt = RepeatedTimer(1, hello, "World")  # it auto-starts, no need of rt.start()

    timer = Direct_fire_timer()
    timer.start(1, helloworld)
    timer.direct_proc()
