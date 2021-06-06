import cv2
import numpy as np 
import glob
orignal = cv2.imread("Images/orignal.jpg")
# =============================================================================
#for loading only single image 
#image_to_compare = cv2.imread("Images/little_darker.jpg")
# 
# =============================================================================

#for loading multiple images
#for f in glob.iglob(r"C:\Users\Rishabh\Desktop\Python In One Video\Project(ImageFetureExtraction)\Programs\Images\*") :

all_images_to_compare = []
titles = []
for f in glob.iglob("Images\*"):
    image = cv2.imread(f)
    all_images_to_compare.append(image)
    titles.append(f)
for image_to_compare, title in zip(all_images_to_compare,titles):
    
            image1=orignal.shape
            #image2=image_to_compare.shape
            
            
            
            
            
            
            # =============================================================================
            # 
            # print(image1) 
            # print(image2)
            # 
            # =============================================================================
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
            orb=cv2.ORB_create()
            #kp_1,descriptor_1=sift.detectAndCompute(orignal,None)
            #kp_2,descriptor_2=sift.detectAndCompute(image_to_compare,None)
            kp_1,descriptor_1=orb.detectAndCompute(orignal,None)
            kp_2,descriptor_2=orb.detectAndCompute(image_to_compare,None)
            print("Keypoints 1st Image : "+ str(len(kp_1)))
            print("Keypoints 2nd Image : "+ str(len(kp_2)))
            
            index_params = dict(algorithm=0,trees=5)
            search_params = dict()
            descriptor_1=np.asarray(descriptor_1,np.float32)
            descriptor_2=np.asarray(descriptor_2,np.float32)
            flann=cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(descriptor_1,descriptor_2,k=2)
            
            
            good_points = []
            
            print("len of matches"+str(len(matches)))
            for m,n in matches:
                if m.distance < 0.6*n.distance:
                    good_points.append(m)
            print("good points length"+str(len(good_points)))
            number_keypoints=0
            if(len(kp_1) >= len(kp_2)):
                number_keypoints = len(kp_1)
            else:
                number_keypoints = len(kp_2)
            
            print("Title : "+title)
            #print(len(matches))
            #print("Good Matches:",len(good_points))
            percentage_similarity =len(good_points)/number_keypoints*100 
            print("similarity : ", str(int(percentage_similarity)) + "%")
            
            #result = cv2.drawMatchesKnn(orignal,kp_1,image_to_compare,kp_2,matches,None)
            
            result=cv2.drawMatches(orignal,kp_1,image_to_compare,kp_2,good_points,None)
            
# =============================================================================
#             cv2.imshow("result",result)
#             
#             cv2.imshow("orignal",orignal)
#             cv2.imshow("image_to_compare",image_to_compare)
# =============================================================================
            print("--------------------------------------")


# =============================================================================
# cv2.waitKey(0)
# cv2.destroyAllWindows();
# 
# =============================================================================
#for resizing we can use this script
#cv2.resize(image_to_compare,None,fx=0.4,fy=0.4)


