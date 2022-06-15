#pylint:disable=no-member
#we will use open cv's bulit in recognizer
import os
import cv2 as cv
import numpy as np

people = ['Bill Gates','Elon Musk','Jeff Bezoz','Kate Middleton','Leonardo Dicaprio','Taylor Swift']
DIR = r'C:\Users\hp\Desktop\photosTrain2\Faces\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')
# for i in os.listdir(r'C:\Users\hp\Desktop\photosTrain'):
#     p.append(i)
# print(p)
features = []#features of the face
labels = []#whose face is it, numerical value basically index
#a function to travserse over every pic stored in the file

def create_train():
    for person in people:#assining the folder
        path = os.path.join(DIR, person)#Use of os.path.join() method to join various path components 
        label = people.index(person)#index of person in people list

        #We are in the folder, now we have to loop over every pic
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')
#print(f'Length of the features ={len(features)}')
#print(f'Length of the labels ={len(labels)}')
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
