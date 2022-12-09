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
We took two identical unconditional DCGANs [[4]](https://arxiv.org/pdf/1511.06434.pdf) and initialized them with random weights from two different seeds. We call them GAN A and GAN B. This ensures that the two architectures start the training process from different points. We train these two GANs on the Fashion-MNIST dataset [[5]](https://github.com/zalandoresearch/fashion-mnist) until convergence. The Fashion-MNIST dataset consists of 60,000 training images of 10 different clothing classes. All hyperparameters are kept the same between the two training process. After training is complete, we unroll both of the networks and compare each feature map from GAN A to each feature map of GAN B and calcualte a similarity score (between 0 and 1). Then we create a similarity score table. The whole process is shown in figure 1.


<p align="center">
  <img src="https://github.com/faderani/DCGAN-Similarity/blob/main/assets/method.gif" alt>
  <br>
  Figure 1: The overview of our method.
</p>


### Results/Insights

Some explanations about similarity matrix, different epochs

<p align="center">
  <img src="https://github.com/faderani/DCGAN-Similarity/blob/main/assets/smiliarity.png" alt>
  <br>
  Figure 2: The overview of our method.
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
