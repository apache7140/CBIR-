import cv2
import numpy as np 
import glob

def siftFeatureDescriptor():
        all_images_to_compare = []
        titles = []
        for f in glob.iglob("Images\*"):
            image = cv2.imread(f)
            all_images_to_compare.append(image)
            titles.append(f)
        for image_to_compare, title in zip(all_images_to_compare,titles):
            
                    image1=orignal.shape
                    # 1) check if 2 images are equal
                    if orignal.shape==image_to_compare.shape:
                        difference = cv2.subtract(orignal,image_to_compare)
                        b,g,r=cv2.split(difference)
                        # cv2.imshow("difference",difference)
                    
                        print(cv2.countNonZero(b))
                        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) ==0 :
                            print("images are completely equal")
                        else:
                            print("they are different")
                    # 2)check for similarities between the 2 images
                            
                    sift=cv2.xfeatures2d.SIFT_create()
                    
                    kp_1,descriptor_1=sift.detectAndCompute(orignal,None)
                    kp_2,descriptor_2=sift.detectAndCompute(image_to_compare,None)
                    print(descriptor_1)
                    print("Keypoints 1st Image : "+ str(len(kp_1)))
                    print("Descriptor of Image"+str(len(descriptor_1)))
                    print("Keypoints 2nd Image : "+ str(len(kp_2)))
                    
                    index_params = dict(algorithm=0,trees=5)
                    search_params = dict()
                    
                    flann=cv2.FlannBasedMatcher(index_params,search_params)
                    matches = flann.knnMatch(descriptor_1,descriptor_2,k=2)
                    
                    
                    good_points = []
                    
                    
                    for m,n in matches:
                        if m.distance < 0.6*n.distance:
                            good_points.append(m)
                    number_keypoints=0
                    if(len(kp_1) >= len(kp_2)):
                        number_keypoints = len(kp_1)
                    else:
                        number_keypoints = len(kp_2)
                    
                    print("Title : "+title)
                    percentage_similarity =len(good_points)/number_keypoints*100 
                    print("similarity : ", str(int(percentage_similarity)) + "%")
                    #result=cv2.drawMatches(orignal,kp_1,image_to_compare,kp_2,good_points,None)
                    print("--------------------------------------")
siftFeatureDescriptor()
#
# def HogFeatureDescriptor():
#     orignal = cv2.imread("Images/orignal.jpg")
#     hog = cv2.HOGDescriptor()
#     h=hog.compute(orignal)
#     frame = cv2.drawKeypoints(orignal,h)
#     cv2.imshow("image",frame)
#
# HogFeatureDescriptor();