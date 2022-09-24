# Python-Neural-Networks

## Overview
The two tasks focused on creating multiple neutral networks and using different architectures and methods. The networks were trained using back propagation and/or front propagation and were parameterizable.

### Task 1
Network 1\
The neural network is built using 2 hidden layers and one output layer with Sigmoid activation functions. As the widths are getting shorter it simulates dimensionality reduction. The network was trained with aa learning rate of 1e-5 over 300 iterations.

![image](https://user-images.githubusercontent.com/76526590/192114590-df68bb0a-55ce-4892-93b6-50171bf4861a.png)

Based on the graphs generated the network initially starts with a low accuracy, but is able to learn how to label the correct item exponentially quickly within the first 50 epochs rising from 10% accuracy and starts to level out from 50 to 300 which is common for most neural networks of this kind. According to figure Trouser was the most accurately labelled where ass the shirt was least accurately. I most commonly being mistaken for the pullovers, mostly probably and mistaking T-shirt for Shirt again probably due to their similar appearances, other methods may be better at dealing with spareness and fine-grained classes. This seems like a common case example as scandals are also commonly mistaken for sneakers, this suggests similar looking items we commonly mistaken. One of the benefits of using this method is the dramatic reduction in the size of the data, over time requiring less space over time.\

Figure 1:

![image](https://user-images.githubusercontent.com/76526590/192114622-736db3cf-7a8a-432c-965d-4d958ee8a0b4.png)

Network 2\
For the second network, the effects of changing the activation function from sigmoid to relu was observed. The architecture remained relatively the same as the only change was sigmoid to relu. We can see that relu learns at a much higher rate than sigmoid so even after lowering the run rate from an accuracy of 78% sooner was accomplished than before (on iteration 271 rather than 291). This seems to be a common occurrence as two of the largest benefits of relu over sigmoid is the reduced likelihood of gradient to vanish – where sigmoid showed very diminished returns accuracy by the 50th epoch, relu showed constant gradient through leading to the faster learning mentioned above. The other benefit of relu is sparsity, sigmoid is more likely to generate non-zero value in dense representations. Though accuracy loss starts to increase both for training and test set, which indicates we may need an even lower learning rate to continue further. Looking at figure 2, we can overserve greater number of values very close to zero for the relu function than the sigmoid function.

![image](https://user-images.githubusercontent.com/76526590/192114956-b4560033-66ed-454a-8c4e-91332f3103a7.png)

Figure 2:

![image](https://user-images.githubusercontent.com/76526590/192114963-8fae04eb-008b-4f9d-846d-a5ed459d6c34.png)

Network 3\
In the third model, relu is continued to be used but the width increases instead of becoming smaller (simulating feature engineering). We see that the width is increasing and needs to trained for a longer period of time before reaching the same accuracy as the previous networks. Based on the graphs below we see losses to be fairly high for a large period of time and remains that way till around the 50th epoch – for which the other networks had already reach nearly their peak accuracy. It peaks around 70% accuracy, being one of the lowest and slowest in terms of raw performance.

![image](https://user-images.githubusercontent.com/76526590/192114984-3b2d5184-a09c-476b-ab9a-78f7368636dc.png)

This network also succumbs to the same issue of fine-grain data issues as the first network where similar looking items were mistaken for each other. The table below we can observe shirts being mistaken for dresses, pullovers and coats. Initially this would suggest this network is the worst option out of the three based on experimental results alone, however, based on more theoretical understanding it may be better in other cases. For example, this network would perform well when there are a lot of empty values, values very close to the correct answer could be substituted in which the network could learn from e.g. potential historical records. The network in our example may have tried to implement this and overshot by grouping the shirt with the other clothing – mitigated by growing some classes together.

![image](https://user-images.githubusercontent.com/76526590/192114989-b97effb1-e02a-4ffd-91cb-9f583aaa1a80.png)

###Task 2

In this task CNN and dropout is looked method are looked at. Pytorch is used instead of creating the network from scratch. The networks varied in performance, with some networks being fairly accurate and other being just better than a coin flip. It should also be noted, if the network couldn’t learn sufficiently in time, the accuracy would be of chance – 1 in 10 (10%).

On the first set of hyperparameters the widths are getting smaller and there is no implementation of any regularisation. It can be observed that training with a learning rate of 1e-2 over 20 iterations we get a test accuracy of 63.1% but it's clear that the test loss is increasing as the training loss is decreasing, implying the network is overfitting.

![image](https://user-images.githubusercontent.com/76526590/192115016-89303c84-3239-439b-aac0-458cc1ca55e4.png)

For the second network everything is left the same apart from introducing Dropout and L2 regularisation. Based on observation of the outcome the learning has slowed down and does not meet the accuracy as before with the same learning rate even over 200 iterations. The initial overfitting of data is dealt with, through the introduction of Dropout. Furthermore, the model becomes more robust. However, the trade-off of implementing dropout is sparse activation. This would work against the L2 regularisation as unlike L1, the former prefers data non spare coefficient vectors. This would have probably led to the worse results from the second network. Furthermore, another disadvantage of dropout would be fallbacks of either having a dropout rate that is too high or too low both resulting in quickly diminishing improvements and observable improvements requiring large number of iterations. This again is seen in the test results as the accuracy is lower and the convergence rate is significantly slower.

![image](https://user-images.githubusercontent.com/76526590/192115023-ad036c1f-47ed-44e1-822b-15ba1b69f339.png)

On the third network the regularisation is slightly lowered, the activation function is changed along with the widths. These changes yielded the highest accuracy out of the three networks. The network reaches accuracy of 65% over 40 epochs making it more accurate than the second network and having a small number iterations like the first network. This is the optimum solution as it combines the best aspect from both the prior two networks and tries to mitigate the downfalls. It deals with the overfitting issue of the first network by using Dropout and regularisation, ensuring the accuracy was higher. It also, deals with the issues of the second network by easing on the regularisation that was causing the Dropout to yield not optimal results as non-sparse coefficients are reduced.

![image](https://user-images.githubusercontent.com/76526590/192115029-94f2f8f8-89bc-4ba0-8da1-746732c741ed.png)

In conclusion, this report has looked at many different types of networks and their implementations. From the results it can be concluded that no particular method is inherently better than the other ones as there were pros and cons for each of them. Task 1 focused on the optimisation of sigmoid and Relu functions, while task 2, focused on parameterization of certain factors, from all these tests the network which had aspects of all the factors performed the best.
