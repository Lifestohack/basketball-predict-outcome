import cv2 as cv
import numpy as np
class BackGroundSubtractor:
	# When constructing background subtractor, we
	# take in two arguments:
	# 1) alpha: The background learning factor, its value should
	# be between 0 and 1. The higher the value, the more quickly
	# your program learns the changes in the background. Therefore, 
	# for a static background use a lower value, like 0.001. But if 
	# your background has moving trees and stuff, use a higher value,
	# maybe start with 0.01.
	# 2) firstFrame: This is the first frame from the video/webcam.
	def __init__(self,alpha,firstFrame):
		self.alpha  = alpha
		self.backGroundModel = firstFrame

	def getForeground(self,frame):
		# apply the background averaging formula:
		# NEW_BACKGROUND = CURRENT_FRAME * ALPHA + OLD_BACKGROUND * (1 - APLHA)
		self.backGroundModel =  frame * self.alpha + self.backGroundModel * (1 - self.alpha)

		# after the previous operation, the dtype of
		# self.backGroundModel will be changed to a float type
		# therefore we do not pass it to cv2.absdiff directly,
		# instead we acquire a copy of it in the uint8 dtype
		# and pass that to absdiff.

		return cv.absdiff(self.backGroundModel.astype(np.uint8),frame)