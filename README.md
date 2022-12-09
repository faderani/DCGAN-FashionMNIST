# Feature Matching and Similarity in GANs
## Northwestern University CS496: Deep Generative Models (Prof. Bryan Pardo)
#### By: Amil Dravid, Soroush Shahi
#### amildravid2023@u.northwestern.edu, soroush@u.northwestern.edu
## 

<p align="center">
  <img src="https://github.com/faderani/DCGAN-Similarity/blob/main/assets/generation.gif" alt>
<br>
  50 random images generated by DCGAN at each epoch during training. 
</p>


### Introduction/Motivation

The performance of deep learning in vision tasks such as image classification and generation has reached new heights in the past decade. However, the internal mechanics of these models remains an open problem[ [1]](https://arxiv.org/pdf/1905.00414.pdf). Visualizing and understanding deep neural nets can assist in the development of better models and new insights[ [2]](https://openreview.net/pdf?id=Hyg_X2C5FX). To this end, one interesting pattern to look at is the similar learned representaions in generative adversarial networks (GANs)[ [3]](https://arxiv.org/pdf/1406.2661.pdf). Finding similar representations can potentially enable downstream applications such as distilling knowledge for more efficient GANs or editing images. In this work, we propose a method to find the similar units (i.e, channels) in DCGANs by calculating a similarity score table. 

### Method
We took two identical unconditional DCGANs [[4]](https://arxiv.org/pdf/1511.06434.pdf) and initialized them with random weights from two different seeds. We call them GAN A and GAN B. This ensures that the two architectures start the training process from different points. We train these two GANs on the Fashion-MNIST dataset [[5]](https://github.com/zalandoresearch/fashion-mnist) until convergence. The Fashion-MNIST dataset consists of 60,000 training images of 10 different clothing classes. All hyperparameters are kept the same between the two training process. After training is complete, we unroll both of the networks and compare each feature map from GAN A to each feature map of GAN B and calculate a similarity score (between 0 and 1). Details of calculate can be found in the code. Then we create a similarity score table. The whole process is shown and explained in Figure 1.


<p align="center">
  <img src="https://github.com/faderani/DCGAN-Similarity/blob/main/assets/method.gif" alt>
  <br>
  Figure 1: The overview of our method. After training the GANs, we use 10000 random z vectors and feed them through both the GANs, collecting the responses of the channels or units as the GANs generate the images. For each unit in one GAN, we calculate similarity with all units in the other GANs, and average over all 10000 samples. 
</p>


### Results/Insights

Below we display the similarity matrices as we train the two GANs across multiple epochs. Brighter colors indicate a better match. Notice how there is not perfect correspondence across the top left to bottom right diagonal, which would indicate each corresponding unit in the two GANs match. This suggests that there is significant stochasticity in training GANs, and the GANs will not necessarily learn the same or similar representation in each unit across different runs. A key insight is that the best similarity occurs deeper into the GAN, when the image is almost generated. Interestingly, there is more similarity in the early layers at the beginning of training, but they diverge across the epochs.

<p align="center">
  <img src="https://github.com/faderani/DCGAN-Similarity/blob/main/assets/smiliarity.png" alt>
  <br>
  Figure 2: Similarity matrices for the two GANs at the beginning of training and at the end. The downsampled similarity matrices are provided for easier visualization. Notice how the early units lose similarity during training, but the later layers obtain the highest similarity. 
</p>

Some explanations about results


<p align="center">
  <img src="https://github.com/faderani/DCGAN-Similarity/blob/main/assets/results.png" alt>
  <br>
  Figure 3: The overview of our method.
</p>

### Future Works


<p align="center">
  <img src="https://github.com/faderani/DCGAN-Similarity/blob/main/assets/match_diff.png" width="340" alt>
  <br>
  Figure 4: The overview of our method.
</p>
