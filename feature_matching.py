import cv2
import numpy as np 

img1=cv2.imread("C:/Users/Rishabh/Pictures/Camera Roll/WIN_20200502_005310.JPG", cv2.IMREAD_GRAYSCALE)

img2=cv2.imread("C:/Users/Rishabh/Pictures/Camera Roll/WIN_20200502_005317.JPG", cv2.IMREAD_GRAYSCALE)

#ORB Detector
'''
cv2.imshow("img",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

orb= cv2.ORB_create()

kp1,des1 = orb.detectAndCompute(img1,None)

kp2,des2 = orb.detectAndCompute(img2,None)

for d in des1:
 	print(d)
	
#Brute Force Matcher
# =============================================================================
# 
# bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
# matches = bf.match(des1,des2)
# matches = sorted(matches, key=lambda x:x.distance) 
# 
# for m in matches:
# 	print(m.distance)
# matching_result=cv2.drawMatches(img1,kp1,img2,kp2,matches[40:80],None)
# 
# 
# cv2.imshow("img",img1)
# cv2.imshow("img2",img2)
# cv2.imshow("matching_result",matching_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 
# =============================================================================
