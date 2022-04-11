import cv2 as cv
import mediapipe as mp
import numpy as np
import bash_info as sh


class HandDetection(object):
  """ Hand detection class with MediaPipe """

  def __init__(self, min_detection_confidence,
               min_tracking_confidence):
    self.__kMpDrawing = mp.solutions.drawing_utils
    self.__kMpHands = mp.solutions.hands
    self.__kMinDetectionConfidence = min_detection_confidence
    self.__kMinTrackingConfidence = min_tracking_confidence
    self.__landmark_position_hand_1 = np.zeros((21, 2), dtype=int)  # lm_id, x, y
    self.__landmark_position_hand_2 = np.zeros((21, 2), dtype=int)  # lm_id, x, y
    self.__height = 0
    self.__width = 0
    self.__results = 0
    self.drawHands = False
    sh.success('HandDetection initialized')

  def __str__(self):
    return 'HandDetection object with min_detection_confidence: {} and min_tracking_confidence: {} .'.format(
      self.__kMinDetectionConfidence, self.__kMinTrackingConfidence)

  def DetectLandmarks(self, img_bgr):
    """
    :param img_bgr:
    """
    with self.__kMpHands.Hands(
      static_image_mode=False,  # Video Stream
      max_num_hands=2, min_detection_confidence=self.__kMinDetectionConfidence,
      min_tracking_confidence=self.__kMinTrackingConfidence) as hands:
        self.__height, self.__width, channel = img_bgr.shape
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        img_rgb = cv.flip(img_rgb, 1)
        self.__results = hands.process(img_rgb)

  def GetLandmarkPositions(self, kHandNumber, img=None):
    """
    :param img: img
    :param kHandNumber: Number of the hand which is wanted to work
    :returns: The pixel positions of the hand landmarks
    """
    if self.__results.multi_hand_landmarks:
      for num, hand in enumerate(self.__results.multi_hand_landmarks):
        if num == 0:
          for id_num, lm in enumerate(hand.landmark):
            self.__landmark_position_hand_1[id_num] = [int(lm.x * self.__width), int(lm.y * self.__height)]
            if self.drawHands:
              self.__kMpDrawing.draw_landmarks(img, hand, self.__kMpHands.HAND_CONNECTIONS)
        elif num == 1:
          for id_num, lm in enumerate(hand.landmark):
            self.__landmark_position_hand_2[id_num] = [int(lm.x * self.__width), int(lm.y * self.__height)]
        else:
          sh.warning('More than 2 hands in visual')
    else:
      sh.error('No hands in visual')
    if kHandNumber == 1:
      return self.__landmark_position_hand_1
    elif kHandNumber == 2:
      return self.__landmark_position_hand_2
    else:
      return sh.warning('Wrong argument')


def main():
  try:
    hd = HandDetection(0.8, 0.5)
    cap = cv.VideoCapture(0)
    while cap.isOpened():
      success, img = cap.read()
      hd.drawHands = True
      hd.DetectLandmarks(img)
      print(hd.GetLandmarkPositions(1, img))
      cv.imshow('frame', img)
      if cv.waitKey(10) & 0xFF == ord('q'):
        break
    cap.release()
    cv.destroyAllWindows()
  except KeyboardInterrupt:
    sh.warning("Keyboard Interrupt! Exiting script")


if __name__ == '__main__':
  main()
