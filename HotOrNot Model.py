#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image

import os


# In[ ]:


from tensorflow.keras import regularizers


# In[23]:


cv2.imread("F:/HotOrNot/Training/Good/0004.jpg").shape


# ## train = ImageDataGenerator(1/255)
# validation = ImageDataGenerator(1/255)

# In[24]:


train = ImageDataGenerator(1/255)
validation = ImageDataGenerator(1/255)


# In[5]:


train_dataset = train.flow_from_directory('F:/HotOrNot/Training/',
                                        target_size= (150,150),
                                         batch_size = 50,
                                         class_mode ='binary')

validation_dataset = train.flow_from_directory('F:/HotOrNot/Validation',
                                        target_size= (150,150),
                                         batch_size = 50,
                                         class_mode ='binary')


# In[6]:


train_dataset.class_indices


# In[35]:


# Download Transfer Learning
#from tensorflow.keras.applications.inception_v3 import InceptionV3


# In[38]:


#TRANSFER LEARNING WITH INCEPTIONV3
base_model =  InceptionV3(weights='imagenet', include_top= False)

x= base_model.output
x =tf.keras.layers.GlobalAveragePooling2D()(x)
x= tf.keras.layers.Dense(512,activation = 'relu')(x)
x= tf.keras.layers.BatchNormalization()(x)

Predictions=tf.keras.layers.Dense(1,activation = 'sigmoid')(x)

for layer in base_model.layers:
    layer.trainable = False #Freeze the layers not to train
    
#Creating Final MOdel    

model = tf.keras.models.Model(inputs = base_model.inputs,outputs=Predictions )


# In[7]:


#model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (150,150,3)),
 #                                   tf.keras.layers.MaxPool2D(2,2),
  #                                  #
   #                                 tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', input_shape = (150,150,3)),
    #                                tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    #tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', input_shape = (150,150,3)),
                                    #tf.keras.layers.MaxPool2D(2,2),
                                    ##
     #                               tf.keras.layers.Flatten(),
                                    ##
      #                              tf.keras.layers.Dense(512,activation = 'relu',kernel_regularizer='l2'),
                                    ##
       #                             tf.keras.layers.Dense(1,activation = 'sigmoid')
#                                   ])


# In[39]:



from tensorflow.keras.optimizers import Adam
model.compile(loss = 'binary_crossentropy',
             optimizer = Adam(learning_rate=0.001),
             metrics =['accuracy'])


# In[41]:


model_fit = model.fit(train_dataset,
                     steps_per_epoch = len(train_dataset),
                    epochs = 5,
                    validation_data = validation_dataset)


# In[42]:


dir_path = 'F:/HotOrNot/test/'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+ i,target_size =(150,150))
    plt.imshow(img)
    plt.show()
    
    x = image.img_to_array(img)
    x= np.expand_dims(x, axis= 0)
    test_images = np.vstack([x])
    
    val=model.predict(test_images)
    print(val)
    if val < 0.75:
        print("Average")
    else:
        print("Good")


# In[43]:


model.summary()


# In[45]:


import pickle


# In[ ]:





# In[ ]:




