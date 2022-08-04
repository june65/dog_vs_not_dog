import numpy as np
import tensorflow as tf
import keras
import glob

model = keras.models.load_model('./dog_vs_not_dog_v3')
dog_images = glob.glob('.\\data\\dog_test_data\\*')
score = [0,0]
i = 0
for image in dog_images:

    img=tf.keras.preprocessing.image.load_img(image,target_size=(300, 300))
    x=tf.keras.preprocessing.image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images)
    if classes[0] == 1 :
        score[0] += 1    
    else :
        score[1] += 1
    #i += 1
    #if i==1000:
    #   break

print(score)