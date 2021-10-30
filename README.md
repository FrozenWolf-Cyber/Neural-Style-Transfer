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

