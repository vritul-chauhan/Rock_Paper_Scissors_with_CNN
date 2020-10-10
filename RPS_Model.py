#!/usr/bin/env python
# coding: utf-8

# # Rock Paper Scissor

# ## Importing libraries

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


btch = 32


# ## Data Preprocessing

# ### Training Data

# In[3]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode = 'wrap',
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 96),
        color_mode='grayscale',
        batch_size=btch,
        class_mode='categorical')


# ### Test Data

# In[4]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 96),
        color_mode='grayscale',
        batch_size=btch,
        class_mode='categorical')


# ## Building the CNN

# ### Initialize the CNN

# In[5]:


cnn = tf.keras.models.Sequential()


# ### Adding Hidden Layer 1

# In[6]:


cnn.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, activation = 'relu', input_shape = [64,96,1]))


# ### Adding Max Pool Layer 1

# In[7]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))


# ### Adding Hidden Layer 2

# In[8]:


cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))


# ### Adding Max Pool Layer 2

# In[9]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2))


# ### Hidden Layer 3 and Max Pool Layer 3

# In[10]:


cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2))


# ### Flattening

# In[11]:


cnn.add(tf.keras.layers.Flatten())


# ### Fully Connected Layer 

# In[12]:


cnn.add(tf.keras.layers.Dense(units = 512, activation = 'relu'))


# ### Output Layer

# In[13]:


cnn.add(tf.keras.layers.Dense(units = 3, activation = 'softmax'))


# In[14]:


cnn.summary()


# ## Compiling the Neural Net

# In[15]:


cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Training the Neural Net

# In[31]:


cnn.fit(x = training_set, validation_data = test_set, steps_per_epoch = 78, epochs = 35, validation_steps = 372/32)


# ### Evaluating Neural Net on Test set

# In[17]:


score = cnn.evaluate(test_set, batch_size = 8, steps = 20)
print()
print('Test accuracy: ', score[1])


# ## Serialize model to JSON

# In[36]:


model_json = cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# ## Serialize weights to HDF5

# In[37]:


cnn.save_weights("model.h5")
print("Saved model to disk")


# In[35]:


import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
test_image = image.load_img('dataset/validation/scissors1.png',color_mode='grayscale' ,target_size = (64, 96))
plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
num = result.argmax()
if num == 0:
    prediction='paper'
elif num==1:
    prediction='rock'
else:
    prediction='scissor'
print(num)


# In[26]:


test_set.class_indices


# In[ ]:




