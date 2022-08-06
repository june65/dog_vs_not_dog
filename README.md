# dog_vs_not_dog
**Image Classification**

|<img src=https://user-images.githubusercontent.com/77375383/182120945-bcd622ff-5320-4389-aa82-8b6b7222d780.jpg></img>|<img src=https://user-images.githubusercontent.com/77375383/182120310-b28772c6-55b7-442d-981c-726c845536d5.jpg></img>|<img src=https://user-images.githubusercontent.com/77375383/182121544-d23fe6c1-9426-4062-8935-66ad3f026954.jpg></img>|<img src=https://user-images.githubusercontent.com/77375383/182121734-69315f6c-97a6-45a6-9a5f-751afcdf8b3b.jpg></img>|
|:-----------:|:-----------:|:-----------:|:-----------:|
| **dog**| **not dog** | **dog** | **not dog** |

+ **Model version**

1. Model_v1
    
    + Train_AC = 0.821 => Test_AC (val_accuracy) = 0.700 
    + because of the low epochs
    
    ![캡처](https://user-images.githubusercontent.com/77375383/182327156-3cbd37c0-ef2f-4a77-80b6-223710a5de05.PNG)
    
2. Model_v2

    + Train_AC = 0.944 => Test_AC (val_accuracy) = 0.851 
    + because of the Overfitting

    ![캡처](https://user-images.githubusercontent.com/77375383/182338225-2e0a0369-6c17-408e-bf06-eb095f5b122b.PNG)
    ![캡처](https://user-images.githubusercontent.com/77375383/182340439-b7ae2797-318f-4ece-b44d-c059108683b5.PNG)
3. Model_v3

    + Train_AC = 0.892 => Test_AC (val_accuracy) = 0.973
    + Add Image augmentation to solve the Overfitting
    
    ![캡처](https://user-images.githubusercontent.com/77375383/182741557-53e36367-5795-4686-9a3b-ce6d111dbb19.PNG)

3. Model_v4

    + Train_AC = 0.792 => Test_AC (val_accuracy) = 0.992
    + larger epoch(1000)
    
    ![image](https://user-images.githubusercontent.com/77375383/183241734-968cf956-a8ce-44be-9ef3-45836a125306.png)
----------------------------------------



+ **Train Dataset**

  + http://vision.stanford.edu/aditya86/ImageNetDogs/
  + https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set
  + https://www.kaggle.com/datasets/venktesh/person-images
  + https://www.kaggle.com/datasets/hartman/dog-breed-identification
  
  + https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
  + https://www.kaggle.com/datasets/ahmadahmadzada/images2000
  + https://www.kaggle.com/datasets/dansbecker/food-101/code
  + https://www.kaggle.com/datasets/aishrules25/automatic-image-captioning-for-visually-impaired
  + https://www.kaggle.com/datasets/aman2000jaiswal/agriculture-crop-images
  + https://www.hemanthdv.org/officeHomeDataset.html
  + https://www.kaggle.com/datasets/andrewmvd/animal-faces
+ **Test Dataset**
  + https://www.kaggle.com/datasets/hartman/dog-breed-identification (train_data)
  

+ **Reference**

  + Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.

  + J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009.
