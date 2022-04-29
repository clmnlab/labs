
"""
component about listening file

In watchdog library, The code can listen file system's event
so using that feature, I made module for listening interest file when files are created
"""

from watchdog.observers import Observer
from watchdog.events import RegexMatchingEventHandler

class CreateFileWatcher:
    def __init__(self, src_path, patterns, listen_file_names=[]):
        self._src_path = src_path
        self._event_handler = RegexMatchingEventHandler(patterns, [], False, True)
        self._event_handler.on_created = self.on_created
        self._event_observer = Observer()
        self._listen_file_names = listen_file_names
        self._is_created = [False] * len(listen_file_names)

    def run(self):
        """
        파일을 기다림
        """

        """
        listening file
        """
        self.start()
        self._event_observer.join()

    def start(self):
        """
        listening file
        """
        self._schedule()
        self._event_observer.start()

    def _schedule(self):
        """
        defien scheduler
        """
        self._event_observer.schedule(
            self._event_handler,
            self._src_path,
            recursive=False
        )

    def on_created(self, event):
        """
        this function is called when a file is created
        """
        from sys import platform
        deliminator = ''
        if platform == 'win32':
            deliminator = '\\'
        else:
            deliminator = '/'

        file_name = event.src_path.split(deliminator)[-1]

        for i in range(0, len(self._listen_file_names)):
            if file_name == self._listen_file_names[i]:
                self._is_created[i] = True

            if sum(self._is_created) == len(self._listen_file_names):
                self._event_observer.stop() # stop event observer if all file concerned my interest is created
                break


if __name__ == "__main__":
    # use this module like this
    watcher = CreateFileWatcher(src_path="/directory_path",
                                patterns=['.*jpg'],
                                listen_file_names=["file_name"])
    watcher.run() # if all my interesting files are not created, process is looping this line
    