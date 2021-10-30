# Neural-Style-Transfer
Comparing perfomance of VGG16 against Alexnet in neural style transfer



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

## Results :

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; AlexNet &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; VGG16

![ezgif-2-57c2378892ca](https://user-images.githubusercontent.com/57902078/139524199-676b84e3-69cf-4611-a2d9-272f6c093372.gif)![ezgif-2-95f1f5582e59](https://user-images.githubusercontent.com/57902078/139524328-27fb7c91-28bd-4d00-9291-788df759bd8a.gif)


![ezgif-2-94c38b1bf951](https://user-images.githubusercontent.com/57902078/139524201-1626449e-6ea6-4b57-afab-120dbc6c8654.gif)![ezgif-2-fbebdef909ca](https://user-images.githubusercontent.com/57902078/139524268-5e914c97-7665-4c41-9ac4-478d3ce49d7f.gif)

![ezgif-2-512abe8f5d94](https://user-images.githubusercontent.com/57902078/139524206-6433877b-df06-4085-8091-31b86111a624.gif)![dance_picasso_gif](https://user-images.githubusercontent.com/57902078/139524506-91edadae-88b2-4edb-8b9d-f05508f90c65.gif)


![ezgif-2-b32404c04deb](https://user-images.githubusercontent.com/57902078/139524208-8992d87d-11fa-4f48-86c2-031a6419574c.gif)![dance_rainbow_gif](https://user-images.githubusercontent.com/57902078/139524258-ac15803b-0d1d-4d64-9b6e-44c4eb71c0f4.gif)

![ezgif-2-bbf492d27c40](https://user-images.githubusercontent.com/57902078/139524211-ffa2e4fe-730c-4cf9-952a-4be515c74141.gif)![ezgif-2-fe6eb508d0e6](https://user-images.githubusercontent.com/57902078/139524281-00046890-4c7b-4be9-b482-8bb864d9a8d0.gif)

## Conclusion :
As we can see VGG16 model is able to retain more features of content and style images than AlexNet due to its greater depth. This shows us that the model learns more complex features as we increase the depth of the model while also increasing the duration for trainning.

