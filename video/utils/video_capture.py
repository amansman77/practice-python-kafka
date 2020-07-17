import cv2
import queue, threading

## 항상 Last Frame만 읽도록 하기 위함
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self.__reader)
        t.daemon = True
        t.start()
    
    def __reader(self):
        while True:
            status, img_raw = self.cap.read()
            if not status:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except Queue.Empty:
                    pass
            self.q.put(img_raw)
    
    def read(self):
        return self.q.get()
