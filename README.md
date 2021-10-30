# Neural-Style-Transfer
Comparing perfomance of VGG16 against Alexnet in Neural Style Transfer . To learn more about Neural Style Transfer with pytorch click ![here](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)


## AlexNet : 
![alexnet](https://user-images.githubusercontent.com/57902078/139523818-6cdf461c-8919-45bd-9ae2-6970f1104f60.png)

 - The AlexNet architecture was introduced in 2012 at the ImageNet Large Scale Visual Recognition Challenge.
 - It was designed by Alex Krizhevsky and published with Illya Sutskever and Krizhevsky’s doctoral advisor Dr. Geoffrey Hinton.
 - AlexNet consisted of 8 layers and used the ReLu activation function which was a major discovery in deep learning. It got rid of the vanishing gradient problem since now the gradient values were not limited to a certain range.
  - It was the first GPU based CNN model and was 4 times faster than previous models.

## VGG16 :
![vgg](https://user-images.githubusercontent.com/57902078/139523822-25552986-6b8f-4447-9b97-70b7fabf252f.jpeg)

- Visual Geometric Group or VGG is a CNN architecture that was introduced 2 years after AlexNet in 2014. The main reason for introducing this model was to see the effect of depth on accuracy while training models for image classification/recognition.
- The VGG network introduced the concept of grouping multiple convolution layers with smaller kernel sizes instead of having one Conv layer with a large kernel size. This caused the number of features at the output to reduce and second was including 3 ReLu layers instead of one increasing learning instances. As can be seen from the image above we see the layered structure (grey boxes) followed by a pooling layer(red boxes).

## Images Used :
### Content :
  ![dancing](https://user-images.githubusercontent.com/57902078/139524723-a47b3fff-72db-4fb9-834a-981871b56b0d.jpg)
### Style :
![curvy](https://user-images.githubusercontent.com/57902078/139524842-51838d14-25cc-4b6c-ad68-3149f5430cc7.jpg)
![fresco](https://user-images.githubusercontent.com/57902078/139524872-75514c18-d52a-4cc9-bc8c-f9694445c326.jpg)
![picasso](https://user-images.githubusercontent.com/57902078/139524877-23522031-938e-4d26-953e-c4dacbe5c35f.jpg)
![rgb](https://user-images.githubusercontent.com/57902078/139524878-5b21368c-26bf-4a69-b446-090af3f7d0b3.jpg)
![water_color](https://user-images.githubusercontent.com/57902078/139524883-3b347166-4bd7-4c74-a441-131930c52564.jpg)



## Results :

### VGG16 :
![dance_rainbow_gif](https://user-images.githubusercontent.com/57902078/139524258-ac15803b-0d1d-4d64-9b6e-44c4eb71c0f4.gif)
![ezgif-2-fe6eb508d0e6](https://user-images.githubusercontent.com/57902078/139524281-00046890-4c7b-4be9-b482-8bb864d9a8d0.gif)
![dance_picasso_gif](https://user-images.githubusercontent.com/57902078/139524506-91edadae-88b2-4edb-8b9d-f05508f90c65.gif)
![ezgif-2-95f1f5582e59](https://user-images.githubusercontent.com/57902078/139524328-27fb7c91-28bd-4d00-9291-788df759bd8a.gif)
![ezgif-2-fbebdef909ca](https://user-images.githubusercontent.com/57902078/139524268-5e914c97-7665-4c41-9ac4-478d3ce49d7f.gif)

### AlexNet :
![ezgif-2-b32404c04deb](https://user-images.githubusercontent.com/57902078/139524208-8992d87d-11fa-4f48-86c2-031a6419574c.gif)
![ezgif-2-bbf492d27c40](https://user-images.githubusercontent.com/57902078/139524211-ffa2e4fe-730c-4cf9-952a-4be515c74141.gif)
![ezgif-2-512abe8f5d94](https://user-images.githubusercontent.com/57902078/139524206-6433877b-df06-4085-8091-31b86111a624.gif)
![ezgif-2-57c2378892ca](https://user-images.githubusercontent.com/57902078/139524199-676b84e3-69cf-4611-a2d9-272f6c093372.gif)
![ezgif-2-94c38b1bf951](https://user-images.githubusercontent.com/57902078/139524201-1626449e-6ea6-4b57-afab-120dbc6c8654.gif)



## Conclusion :
As we can see VGG16 model is able to retain more features or information of content and style images than AlexNet due to its greater depth. This shows us that the model learns more complex features as we increase the depth of the model while also increasing the duration for trainning.

