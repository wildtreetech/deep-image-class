from collections import Counter
from threading import Thread
import time

import numpy as np

from sklearn.linear_model import SGDClassifier

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

# Install squeezenet with `pip install keras_squeezenet`
from keras_squeezenet import SqueezeNet

import cv2

image_size = 227


class WebcamVideoStream:
    def __init__(self, src=0):
        """Capture a video stream in a new thread."""
        self.stream = cv2.VideoCapture(src)
        # small hack to make sure camera is ready
        time.sleep(2.5)
        (self.grabbed, frame) = self.stream.read()
        self.frame = cv2.resize(frame, (image_size, image_size),
                                fx=0, fy=0)

        self.stopped = False

    def start(self):
        self._thread = Thread(target=self._loop, args=()).start()
        return self

    def _loop(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, frame) = self.stream.read()
            self.frame = cv2.resize(frame, (image_size, image_size),
                                    fx=0, fy=0)

    def stop(self):
        """Stop the capturing"""
        self.stopped = True

    def read(self):
        """Read the latest captured frame"""
        return self.grabbed, self.frame

    def release(self):
        self.stop()
        time.sleep(0.5)
        return self.stream.release()


class Learner:
    def __init__(self, n_classes=3):
        self._fitted = False
        self.X = []
        self.y = []
        self.X_conv = []
        self._lr = SGDClassifier(loss='log', penalty='l1',
                                 #warm_start=True,
                                 max_iter=50, tol=1e-3, random_state=2+3)
        self._features = SqueezeNet(weights='imagenet',
                                    input_shape=(image_size, image_size, 3))

    def add_image(self, img, klass):
        self.X.append(img)
        self.y.append(klass)
        X = np.array(self.X[-1:], dtype=np.float32)
        X = preprocess_input(X)
        X_conv = self._features.predict(X)
        print(decode_predictions(X_conv))
        X_conv = X_conv[0]
        print(X_conv.shape)
        X_conv = X_conv / np.linalg.norm(X_conv)
        self.X_conv.append(X_conv)
        print(Counter(self.y))

    def fit(self):
        X_conv = np.array(self.X_conv)
        y = np.array(self.y)
        print(X_conv.shape, y.shape)
        if len(set(self.y)) > 1:
            self._lr.fit(X_conv, y)
            self._fitted = True

    def predict(self, img):
        if not self._fitted:
            return 0
        # a batch of one image
        X = np.array(img, dtype=np.float32)
        X = X[np.newaxis]
        X = preprocess_input(X)
        X_conv = self._features.predict(X)[0]
        X_conv = X_conv / np.linalg.norm(X_conv)
        return self._lr.predict([X_conv])[0]


def identify_gesture(frame):
    """Identify the player's gesture from a webcam frame"""
    # Add your algorithm here
    # You will have to:
    # * persist the logistic regression model you trained in the notebook
    # * setup the same pretrained network you use as feature extractor
    # * resize `frame` to be 227x227 pixels large
    # * apply the preprocessing steps for the pretrained network
    # * load your trained logistic regression model here
    #
    #return random.choice(['rock', 'paper', 'scissor'])
    return 'rock'


def main():
    """Run the rock-paper-scissor game forever"""
    model = Learner()
    cap = WebcamVideoStream(0)
    cap.start()

    tick = time.time()
    n = 0

    while True:
        ret, frame = cap.read()
        n += 1
        output = frame.copy()

        if not ret:
            continue

        # most algorithms outside of OpenCV need RGB colour ordering
        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now = time.time()

        # call the function that will identify what gesture is shown
        # this is where the brains of the game is
        # gesture = identify_gesture(frame_)
        gesture2name = {0: 'rock', 1: 'paper', 2: 'scissor'}
        y_pred = model.predict(frame_)
        gesture = gesture2name[y_pred]

        cv2.putText(output, "%s" % gesture,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
                    )
        cv2.putText(output, "%i FPS" % (n / (now - tick)),
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
                    )

        cv2.imshow('Rock, paper, scissors', output)

        pressed_key = cv2.waitKey(1) & 0xFF
        # exit if the user pressed q
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('r'):
            model.add_image(frame_, 0)
        elif pressed_key == ord('p'):
            model.add_image(frame_, 1)
        elif pressed_key == ord('s'):
            model.add_image(frame_, 2)
        elif pressed_key == ord('f'):
            model.fit()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
