import keras

model = keras.models.load_model('./dog_vs_not_dog_v3')
model.save('./h5_model/dog_vs_not_dog_v3.h5')