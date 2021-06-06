import cv2
import numpy as np
import glob
import csv
import uuid 
import os as os
from pathlib import Path
import pandas as pd

class Feature_Extraction_ORB:
    def feature_extraction(self, path, index_path):
        all_images_to_compare = []
        titles = []
        print("Block1")
        for f in glob.iglob(path):
            image = cv2.imread(f)
            #query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
            #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            all_images_to_compare.append(image)
            titles.append(f)
            print(f)
        return all_images_to_compare,titles
    def ArrayToFile(self,title, descriptor):
        uid=uuid.uuid1()
        array_path = desc_path + str(uid) +'.txt'
        try:
            array_path=Path(array_path)
            np.savetxt(array_path,np.array(descriptor), fmt='%.0f',delimiter=",")
        except Exception :
            print(Exception)
        return array_path

    def writingDescriptorToCSV(self,all_images_to_compare,titles):
            print('Going to Write to the index file')
            output = open(index_path, "w")
            for all_images_to_compare_temp, title in zip(all_images_to_compare, titles):
                print("Reached Here")
                orb = cv2.ORB_create()
                kp,descriptors = orb.detectAndCompute(all_images_to_compare_temp, None)
                self.desc_arr_1=descriptors
                number_of_keypoints = str(len(kp))
                array_path=self.ArrayToFile(title,descriptors)
                output.write("%s,%s,%s,\n" % (title,number_of_keypoints,array_path))
                print("Writing done for"+title)
            output.close()
    def queryImage(self, query_path):
        query_image = cv2.imread(query_path)
        # query_image=cv2.cvtColor(query_image,cv2.COLOR_BGR2GRAY)
        cv2.imshow("query_image", query_image)
        cv2.waitKey(0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp, descriptors = orb.detectAndCompute(query_image, None)
        length_kp_query = len(kp)
        print("length_of_query_keypoint:" + str(length_kp_query))
        return descriptors, length_kp_query

    def fileInput(self,query_image_desc,len_kp_query):
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        fileHandle = open('index.csv','r')
        list_title=[]
        list_percentage = []
        for line in fileHandle:
            fields = line.split(',')
            title = fields[0]
            kp_1=int(fields[1])
            array_path=Path(fields[2])
            arr=np.loadtxt(array_path,delimiter=",")
            ex1=np.asarray(arr,np.float32)
            ex2=np.asarray(query_image_desc,np.float32)
            matches = flann.knnMatch(ex1,ex2,k=2)
            good_points = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                     good_points.append(m)
            number_keypoints = 0
            if (kp_1 >= len_kp_query):
                number_keypoints = kp_1
            else:
                number_keypoints = len_kp_query
            print("Title : " + title)
            print("number_of_keyPoints"+str(number_keypoints))
            print('length_of_good_points'+str(len(good_points)))
            #calculation of the Rank
            percentage_similarity = len(good_points) / number_keypoints * 100
            print("similarity : ", str(int(percentage_similarity)) + "%")
            list_title.append(title)
            list_percentage.append(percentage_similarity)
        fileHandle.close()
        return self.convertingRawDataIntoDataFrames(list_title, list_percentage)

    def convertingRawDataIntoDataFrames(self,list_titles,list_percentage):
        d= {'title':list_titles,'percentages':list_percentage}
        df = pd.DataFrame(d)
        df=df.sort_values(by=['percentages'],ascending=False)
        for a in df :
            print(a)
        print(df)
        return "successfull"


    def deleteFiles(self, desc_path):
        filelist = [f for f in os.listdir(desc_path)]
        for f in filelist:
            print(f)
            os.remove(os.path.join(desc_path, f))
            print(f + " deleted!")

#path = "C:/Users/Rishabh/Desktop/CBIR/HOGDescriptor/Images/*.jpg"

desc_path="Descriptor/"
path="Images/*.jpg"
index_path = "index.csv"
query_image_path="Images/100.jpg"
#just remove the comments down below
'''
'''
#initialize the object
f = Feature_Extraction_ORB()
f.deleteFiles(desc_path)
#initialize the object

#f = Feature_Extraction_ORB()

#extracting descriptors from the images
all_images_to_compare,titles=f.feature_extraction(path, index_path)
#putting all of it into a file
f.writingDescriptorToCSV(all_images_to_compare,titles)
#querying image
'''
'''


query_image_desc,len_kp_query=f.queryImage(query_image_path)
abc=f.fileInput(query_image_desc,len_kp_query)
print(abc)
print("Successfull")
