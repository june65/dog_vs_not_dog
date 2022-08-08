import keras

import tensorflow_hub as hub

#model = tf.saved_model.load('./dog_vs_not_dog_v4')
#model = keras.models.load_model('./dog_vs_not_dog_v3')
model = keras.Sequential([
    hub.KerasLayer("./dog_vs_not_dog_v4")
])

model.save('./h5_model/dog_vs_not_dog_v4.h5')