import os
import shutil
import glob
import random
import string
#dog_data PATH
final_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/train_data_set/dog_data'
data1_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data1/images/Images'
data2_test_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data2/test'
data2_train_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data2/train'
data3_test_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data3/dogImages/dogImages/test'
data3_train_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data3/dogImages/dogImages/train'
data4_train_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data5/train'

#test_data_set 
data4_test_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/dog_test_data'

#not_dog_data PATH
final_not_dog_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/train_data_set/not_dog_data'
data5_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data5/animals'
data6_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data6/images'
data7_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data7'
data8_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data8/visuallyimpair/visual_dataset'
data9_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data9/kag2'
data10_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data10/OfficeHomeDataset_10072016/Real World'
data11_path = 'C:/Users/이준명/Desktop/dog_vs_not_dog/data/data11/afhq/train'


#data1
'''
dog_breed_files = glob.glob(data1_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:

        shutil.move(image, final_path)
'''

#data2
'''
dog_breed_files = glob.glob(data2_test_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:

        #random__name
        _LENGTH = 10
        string_pool = string.ascii_lowercase 
        result = ""
        for i in range(_LENGTH) :
            result += random.choice(string_pool)

        image_newname = image.replace('.jpg','') + result +'.jpg'
        os.rename(image,image_newname)
        shutil.move(image_newname, final_path)

dog_breed_files = glob.glob(data2_train_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:
        
        #random__name
        _LENGTH = 10
        string_pool = string.ascii_lowercase 
        result = ""
        for i in range(_LENGTH) :
            result += random.choice(string_pool)

        image_newname = image.replace('.jpg','') + result +'.jpg'
        os.rename(image,image_newname)
        shutil.move(image_newname, final_path)
'''
#data2 error fix
'''
dog_breed_files = glob.glob(final_path+'/*')

for filename in dog_breed_files:
    image_newname = filename.replace('jpg','') +'.jpg'
    os.rename(filename,image_newname)
'''

#data3
'''
dog_breed_files = glob.glob(data3_test_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:

        shutil.move(image, final_path)

dog_breed_files = glob.glob(data3_train_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:

        shutil.move(image, final_path)
'''

#data4
'''
dog_breed_images = glob.glob(data4_train_path+'/*')

for image in dog_breed_images:

    shutil.move(image, final_path)
'''

#train data number
train_data_images = glob.glob(final_path+'/*')
print(len(train_data_images))
#51702 dog_data

#test data number
test_data_images = glob.glob(data4_test_path+'/*')
print(len(test_data_images))
#10357 dog_data

#data5
'''
dog_breed_files = glob.glob(data5_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:

        image_newname = image.replace('.jpg','') + "1" +'.jpg'
        os.rename(image,image_newname)
        shutil.move(image_newname, final_not_dog_path)
'''

#data6
'''
dog_breed_files = glob.glob(data6_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:

        shutil.move(image, final_not_dog_path)
'''

#data7
'''
dog_breed_files = glob.glob(data7_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:

        shutil.move(image, final_not_dog_path)
'''

#data8
'''
dog_breed_files = glob.glob(data8_path+'/*')

for filename in dog_breed_files:

        shutil.move(filename, final_not_dog_path)
'''
'''

#data9

dog_breed_files = glob.glob(data9_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:

        #random__name
        _LENGTH = 10
        string_pool = string.ascii_lowercase 
        result = ""
        for i in range(_LENGTH) :
            result += random.choice(string_pool)

        image_newname = image.replace('.jpg','') + result +'.jpg'
        os.rename(image,image_newname)
        shutil.move(image_newname, final_not_dog_path)
'''


#data10
'''
dog_breed_files = glob.glob(data10_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:

        #random__name
        _LENGTH = 10
        string_pool = string.ascii_lowercase 
        result = ""
        for i in range(_LENGTH) :
            result += random.choice(string_pool)

        image_newname = image.replace('.jpg','') + result +'.jpg'
        os.rename(image,image_newname)
        shutil.move(image_newname, final_not_dog_path)
'''
#data11
'''
dog_breed_files = glob.glob(data11_path+'/*')

for filename in dog_breed_files:

    dog_breed_images = glob.glob(filename+'/*')

    for image in dog_breed_images:

        shutil.move(image, final_not_dog_path)
'''


#train data number
train_not_dog_data_images = glob.glob(final_not_dog_path+'/*')
print(len(train_not_dog_data_images))
#40113 dog_data