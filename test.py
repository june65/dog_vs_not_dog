import numpy as np
import tensorflow as tf
import glob
import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

#model = tf.saved_model.load('./dog_vs_not_dog_v4')
model = tf.keras.models.load_model('./dog_vs_not_dog_v5')
'''model = tf.keras.Sequential([
    hub.KerasLayer("./dog_vs_not_dog_v3")
])
'''

dog_images = glob.glob('.\\data\\test_data_set\\not_dog_data\\*')

error_dataset = []

score = [0,0]
i = 0
for image in dog_images:

    img=tf.keras.preprocessing.image.load_img(image,target_size=(300, 300))
    x=tf.keras.preprocessing.image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, verbose=1)
    print(classes)
    if classes[0] == 1 :
        score[0] += 1    
        None
    else :
        score[1] += 1
        cv2_image = cv2.imread(image, cv2.COLOR_BGR2RGB)    
        cv2_image2 = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) 
        error_dataset.append(cv2_image2)

    i += 1
    
    if i ==100:
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

