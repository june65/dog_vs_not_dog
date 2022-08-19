import numpy as np
import tensorflow as tf
import glob
import tensorflow_hub as hub
import numpy as np
import cv2

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

error_dataset = []
model = tf.keras.Sequential([
    hub.KerasLayer("./dog_vs_not_dog_v8")
])
#model = tf.keras.models.load_model('./dog_face_AI/AI_Model/ear_model/dog_ear_v4')
dog_images_up = glob.glob('.\\data\\not_dog_data\\*')
dog_images_down = glob.glob('.\\data\\dog_data\\*')
score = [0,0]
i = 0
for image in dog_images_up:

    img=tf.keras.preprocessing.image.load_img(image,target_size=(300, 300))
    x=tf.keras.preprocessing.image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images)
    print(classes[0])
    if classes[0] == 1:
        score[0] += 1    
        None
    else :
        score[1] += 1
        cv2_image = cv2.imread(image, cv2.COLOR_BGR2RGB)    
        cv2_image2 = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) 
        error_dataset.append(cv2_image2)
    i += 1
    if i ==1:
        break

for image in dog_images_down:

    img=tf.keras.preprocessing.image.load_img(image,target_size=(300, 300))
    x=tf.keras.preprocessing.image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])
    print(classes[0])
    classes = model.predict(images)
    if classes[0] < 0.5 :
        score[1] += 1
        cv2_image = cv2.imread(image, cv2.COLOR_BGR2RGB)    
        cv2_image2 = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) 
        error_dataset.append(cv2_image2)
        
    else :
        score[0] += 1    
        None
    i += 1
    if i >=1501:
        break
print(score)

nrows = 10
ncols = 10

fig = plt.gcf()
fig.set_size_inches(nrows * 4, ncols * 4)
j = 0
for j in range(len(error_dataset)):

  sp = plt.subplot(nrows,ncols, j + 1)
  sp.axis('Off')
  plt.imshow(error_dataset[j])

plt.show()

