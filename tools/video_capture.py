import cv2

class Cap:
    def __init__(self, path, step_size=1):
        self.path = path
        self.step_size = step_size
        self.curr_frame_no = 0

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.path)
        return self

    def read(self):
        success, frame = self.cap.read()
        if not success:
            return success, frame
        for _ in range(self.step_size):
            s, f = self.cap.read()
            if not s:
                break

        return success, frame

    def read_all(self):
        frames_list = []
        while True:
            success, frame = self.cap.read()
            if not success:
                return frames_list

            frames_list.append(frame)

            for _ in range(self.step_size-1):
                s, f = self.cap.read()
                if not s:
                    return frames_list

    def __exit__(self, a, b, c):
        self.cap.release()
        cv2.destroyAllWindows()