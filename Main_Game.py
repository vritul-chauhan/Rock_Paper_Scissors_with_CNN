#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import time 
from PIL import Image 
import PIL
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import os
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
# load weights into new model
cnn.load_weights("model.h5")
print("Loaded model from disk")
font = cv2.FONT_HERSHEY_SIMPLEX 


# In[2]:


cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
folder_path = 'guess\main'


# In[3]:


def winner(t1, t2):
    if t1 == t2:
        return 0
    elif t1==1 and t2==2:
        return 1
    elif t1==2 and t2==1:
        return 2
    elif t1==0 and t2==1:
        return 1
    elif t1==1 and t2==0:
        return 2
    elif t1==0 and t2==2:
        return 2
    else:
        return 1
    
def janken():
    images = []
    img = image.load_img('left.jpg', color_mode='grayscale', target_size=(64,96))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
    img = image.load_img('right.jpg', color_mode='grayscale', target_size=(64,96))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

    images = np.vstack(images)
    classes = cnn.predict(images, batch_size=2)
    t1=classes[0].argmax()
    t2=classes[1].argmax()
    return winner(t1,t2),t1,t2

def img_split_left():
    left1 = 10
    right1 = 310
    top = 90
    bottom = 470
    imk = Image.open("hand.jpg")
    imk = imk.crop((left1, top, right1, bottom))
    imk.save("left.jpg")
    imk.close()
    
def img_split_right():
    left2 = 330
    right2 = 630
    top = 90
    bottom = 470
    imk = Image.open("hand.jpg")
    imk = imk.crop((left2, top, right2, bottom))
    imk.save("right.jpg")
    imk.close()

def which_pos(num):
    if num==0:
        pl = 'Paper'
    elif num ==1:
        pl = 'Rock'
    else:
        pl = 'Scissors'
    return pl


# In[4]:


# Open the camera 
cap = cv2.VideoCapture(0) 
while True: 
    TIMER = 3
    # Read and display each frame 
    ret, img = cap.read() 
    img = cv2.flip(img,1)
    img = cv2.rectangle(img,(5,85),(315,475),(0,150,0),3)
    img = cv2.rectangle(img,(325,85),(635,475),(0,0,150),3)
    cv2.putText(img, 'Player 1', 
                        (20, 60), font, 
                        2, (0, 255, 255), 
                        2, cv2.LINE_AA)
    cv2.putText(img, 'Player 2', 
                        (340, 60), font, 
                        2, (0, 255, 255), 
                        2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor', img) 
    # check for the key pressed 
    k = cv2.waitKey(125) 
    
    #press ENTER to start the COUNTDOWN
    if k == 13: 
        prev = time.time() 

        while TIMER >= 0: 
            ret, img = cap.read() 
            img = cv2.flip(img,1)
            img = cv2.rectangle(img,(5,85),(315,475),(0,150,0),3)
            img = cv2.rectangle(img,(325,85),(635,475),(0,0,150),3)
            cv2.putText(img, 'Player 1', 
                        (20, 60), font, 
                        2, (0, 255, 255), 
                        2, cv2.LINE_AA)
            cv2.putText(img, 'Player 2', 
                        (340, 60), font, 
                        2, (0, 255, 255), 
                        2, cv2.LINE_AA)
            # Display countdown on each frame 
            # specify the font and draw the 
            # countdown using puttext 
            
            cv2.putText(img, str(TIMER), 
                        (245, 320), font, 
                        7, (0, 255, 255), 
                        4, cv2.LINE_AA) 
            cv2.imshow('Rock Paper Scissor', img) 
            cv2.waitKey(125) 

            # current time 
            cur = time.time() 

            # Update and keep track of Countdown 
            # if time elapsed is one second 
            # than decrese the counter 
            if cur-prev >= 1: 
                prev = cur 
                TIMER = TIMER-1

        else: 
            ret, img = cap.read() 
            img = cv2.flip(img,1)
            img = cv2.rectangle(img,(5,85),(315,475),(0,150,0),3)
            img = cv2.rectangle(img,(325,85),(635,475),(0,0,150),3)
            cv2.putText(img, 'Player 1', 
                        (20, 60), font, 
                        2, (0, 255, 255), 
                        2, cv2.LINE_AA)
            cv2.putText(img, 'Player 2', 
                        (340, 60), font, 
                        2, (0, 255, 255), 
                        2, cv2.LINE_AA)
            
            cv2.imshow('Rock Paper Scissor', img) 

            # time for which image displayed 
            cv2.waitKey(500) 

            # Save the frame 
            cv2.imwrite('hand.jpg',img)
            img_split_left()
            img_split_right()
            num,t1,t2 = janken()
            pl1 = which_pos(t1)
            pl2 = which_pos(t2)
            cv2.putText(img, pl1, 
                    (20, 350), font, 
                    1, (0, 255, 255), 
                    1, cv2.LINE_AA)
            cv2.putText(img, pl2, 
                    (340, 350), font, 
                    1, (0, 255, 255), 
                    1, cv2.LINE_AA)
            if num == 0:
                cv2.putText(img, 'DRAW', 
                    (50, 460), font, 
                    2, (0, 255, 255), 
                    2, cv2.LINE_AA)
            else:
                cv2.putText(img, 'Player '+str(num)+' won!', 
                    (50, 460), font, 
                    2, (0, 255, 255), 
                    2, cv2.LINE_AA)
                print(num)
            cv2.imshow('Rock Paper Scissor', img) 
            cv2.waitKey(1000) 
            cv2.imwrite('result.jpg',img)
            # HERE we can reset the Countdown timer 
            # if we want more Capture without closing 
            # the camera 

    # Press Esc to exit 
    elif k == 27: 
        break

# close the camera
cap.release() 
cv2.destroyAllWindows()


# In[ ]:




