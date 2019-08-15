# import the necessary packages
from traditional_features.rootsift import RootSIFT
import cv2


# load the image we are going to extract descriptors from and convert
# it to grayscale
# image = cv2.imread("example.png")
image = cv2.imread("example1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
# detect Difference of Gaussian keypoints in the image
# detector = cv2.FeatureDetector_create("SIFT")
# kps = detector.detect(gray)

# extract normal SIFT descriptors
# extractor = cv2.DescriptorExtractor_create("SIFT")
(kps, descs) = sift.detectAndCompute(gray, None)
print("SIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape))

# extract RootSIFT descriptors
rs = RootSIFT()
(kps, descs) = rs.compute(gray, kps)
print("RootSIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape))