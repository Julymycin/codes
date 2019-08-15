from LogGabor import LogGabor
import cv2

parameterfile = './lg_para.py'
lg = LogGabor(parameterfile)

image = cv2.imread("example1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lg.set_size(gray)
