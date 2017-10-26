# Quick overview of the state-of-the-art: the classification task of the ImageNet challenge


## Introduction


In the previous post, the interest of embed deep learning algorithms in
mobile phone has be introduced. With diverse and rich photo gallery
everyone has in his or her cellphone, it becomes an important personal
database difficult to manage by ourself specially to recover specific
events.

Comet is an app helping users rediscovering their own souvenirs while
respecting their privacy. As a start, users should be able to have a
souvenir in their mind and find the associated images in the most
efficient way. In a large gallery containing thousands of images, no one
have the exact time line and can exactly remember its structure.
Nevertheless everyone has key events in its mind where we can remember
of tiny details. In this way, users may search a specific image using
its memory of objects present in this targeted picture.

Search images by objects, animals or persons is a classification task
with a high number of classes. For a specific word, a user can be able
to find a picture according to a confidence probability predicted by as
specific model. Thus the higher the number of classes, the better the
match with the vocabulary of the user.

The purpose of this post is to provide few details about the
state-of-the-art of the ImageNet challenge, precisely over the
classification task that have inspired our work at the machine learning
team. Some of the major shifts of performances will be approached
providing short precisions about innovative architectures and modules.

## The classification task using Deep Learning: state-of-the-art using the ImageNet challenge

### The ImageNet challenge

In the recent field of computer vision and image analysis, the ImageNet
challenge is a reference for the state-of-the-art. It is the outcome of
a collaboration between Stanford University and Princeton University
grouping around fourteen millions of images originally labeled with
Synsets[^1] of the WordNet lexicon tree. The original challenge
consisted in a simple classification task, each image belows to one
categorie, with 1000 categories from specific breed of dog to precise
type of food. This task for the ImageNet challenge is still available
but it also has evolved in a multi-classification task while inferring
bounding boxes around the different objects of the image. This second
challenge will not be developed in this post.

![01_carbonara](01_carbonara.JPEG)*Example of image in ImageNet2012 dataset: Carbonara. Source: [ImageNet](www.image-net.org/)*

![02_EnglishFoxhound](02_EnglishFoxhound.JPEG)*Example of image in ImageNet2012 dataset: English Foxhound. Source: [ImageNet](www.image-net.org/)*

### The advent of deep learning

The original classification task of the ImageNet challenge has been the
opportunity for the Deep Learning community to make itself known with an
explosion of the performances using neural networks. Inspired from the
work of [Y. Lecun and al (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), the first deep learning model outperforming the ImageNet challenge published by [A. Krizhevsky and al (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) has made a lot of noise by getting a top-5 error rate of 15.3% when the
previous best one was 26.2%. This famous model called “AlexNet” is a
simple application of five consecutive convolutional filters, max-pool
layers and tree fully-connected layers.

![1_LeNet_5_architecture_digit_reco](11_LeNet_5_architecture_digit_reco.png)*LeNet-5 architecture for digit recognition. Source: [Y. Lecun and al (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)*

![12_AlexNet_architecture_training_2_gpu](12_AlexNet_architecture_training_2_gpu.png)*AlexNet architecture for training with 2 GPUs. Source: [A. Krizhevsky and al (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)*


### Going deeper

Since the 2012 milestone, the active researches on deep learning
algorithms have tried to go deeper in the sequences of convolutional
layers. In the first submission of [K. Simonyan & A. Zisserman (2015](https://arxiv.org/abs/1409.1556.pdf) in 2014, they provide details about the VGG16
model, its performances and its architecture. It is composed of sixteen
convolutional layers, multiple max-pool layers and tree final
fully-connected layers. One of its specificities is to sequence multiple
convolutional layers with ReLu activations creating non-linear
transformations. Moreover the authors justify the 3x3 filters for each
convolution (11x11 filters for the AlexNet) by recognizing same patterns
as larger filters and decreasing the number of parameters to train.
These transformations reached to a 7.3% top-5 score on the 2014 ImageNet
challenge reducing by a factor of two the error rate of the AlexNet.

### Inception modules

New ideas have emerged this same year, [M. Lin and al (2014)](https://arxiv.org/abs/1312.4400.pdf) have developed the concept of
“inception”. Original convolutional layer uses linear transformations
with a non-linear activation function. Train multiple convolutional
layers simultaneously and stack their feature maps linked with a
multi-layer perceptron produces a non-linear transformation over the all
maps. This idea has been exploited by Google researchers [C. Szegedy and al (2014)](https://arxiv.org/abs/1409.4842) to construct a deeper network called
GoogLeNet (or Inception V1) with 22 layers using “Inception modules” for
a total of over 50 convolution layers. Each module is composed of 1x1,
3x3, 5x5 convolution layers and a 3x3 max-pool layer to increase
sparsity in the model and obtain different type of patterns. The feature
maps produced are concatenated and analyzed by the next inception
module. A most complexed module is also applied using 1x1 convolution
before the 3x3 and 5x5 ones, it realizes linear projections (seen as
linear rectifications) and dimension reductions while parameters learnt
focus the patterns over the feature maps. The GoogLeNet model has a 6.7%
error rate over the 2014 ImageNet challenge which is somewhat lower than
the VGG16 but a remark needs to be done: the VGG16 model is around 490MB
and the GoogLeNet is around 55MB. This gap is explained by the three
large fully-connected layers in the VGG architecture.

![31_Inception_module](31_Inception_module.png)*Inception module. Source: [C. Szegedy and al (2014)](https://arxiv.org/abs/1409.4842)*

![32_GoogLeNet](32_GoogLeNet.png)*GoogLeNet architecture. Source: [C. Szegedy and al (2014)](https://arxiv.org/abs/1409.4842)*

In 2015, [C. Szegedy and al (2015)](https://arxiv.org/abs/1512.00567) have once more released a paper and first develop the Inception V2 model. It is mostly inspired from the first
version but the authors has changed the large 5x5 filter in the
inception modules by two 3x3 filters, the fist one is a 3x3 convolution
and the second one is a 3x1 fully-connected slided over the fist one.
This method is called convolution factorization, its decreases the
number of parameters in each inception module and reduce the computation
cost. This Inception V2 model with a new architecture has been tested on
the 2012 ImageNet challenge dataset with a top-5 error rate of 5.6%
which is better than the previous version. The third version called
Inception V3 is also described by [C. Szegedy and al (2015)](https://arxiv.org/abs/1512.00567), the authors have fine
tune the batch-normalization and they have selected a high resolution
input. Notably they have reduced the strides of the first two layers and
removed a max-pool layer to analyze images with higher precision. Hence
they have reached a top-5 error rate of 3.58% over the 2012 ImageNet
challenge.

![33_Inception_module_factorization_after_nxn_conv](33_Inception_module_factorization_after_nxn_conv.png)*Inception module factorization application replacing 5x5 convolution
by two 3x3 convolutions. Source: [C. Szegedy and al (2015)](https://arxiv.org/abs/1512.00567)*

![34_Inception_module_application_factorization_replace_5x5_by_two_3x3](34_Inception_module_application_factorization_replace_5x5_by_two_3x3.png)*Inception module factorization application replacing 5x5 convolution
by two 3x3 convolutions. Source: [C. Szegedy and al (2015)](https://arxiv.org/abs/1512.00567)*


### Residual learning

Almost at the same time of the release of Inception V3, Microsoft has
also developed an interesting idea published by [K. He and al (2015)](http://arxiv.org/abs/1512.03385). As we have seen before,
the main common trend in convolutional neural network models is their
increasing depth. The authors notice that the increasing depth involves
a increasing error rate which it is not due to overfitting but to the
difficulties to train and optimize an extreme deep model. To deal with
this, “Residual Learning” has been introduced to create a connection
between the output of one or multiple convolutional layers and their
original input with an identity mapping. It means that the model is
trying to learn a residual function which keeps most of the information
and produces slightly changes. Thus patterns from the input image can be
learnt in deeper layers because its not too much transformed yet.
Moreover, this method doesn’t add any additional parameter and it
doesn’t increase the computation complexity of the model. The model
presented by [K. He and al (2015)](http://arxiv.org/abs/1512.03385) is usually called “ResNet” and it is composed of
152 convolutional layers with 3x3 filters using residual learning by
block of two layers. Even if it has got a top-5 error rate of 4.49% over
the 2012 ImageNet challenge (less than the Inception V3), the ResNet
model has won the 2015 challenge with a top-5 error rate of 3.57% to be
the new state-of-the-art model.

![41_Residual_learning_block](41_Residual_learning_block.png)*Residual learning block architecture. Source: [K. He and al (2015)](http://arxiv.org/abs/1512.03385)*

![42_VGG19_vs_ResNet_34_layers](42_VGG19_vs_ResNet_34_layers.png)*ResNet architecture. Source: [K. He and al (2015)](http://arxiv.org/abs/1512.03385)*

### The Inception-ResNet

As a response, Google researchers [C. Szegedy and al (2016)](http://arxiv.org/abs/1602.07261) have published one year
after the success of the ResNet model. The objective was to combine
residual connections from ResNet and the architecture of Inception V3
using inception modules. Building residual inception blocks, the
Inception V4 (Inception-ResNet)[^2] can be trained faster and it
outperforms the state-of-the-art over the 2012 ImageNet challenge. It
combines inception modules to increase sparsity and residual blocks to
learn deeper layers. The inception modules have been improved to
fine-tune the layer sizes and with complex architectures in order to
detect specific patterns. This model also omits batch-normalization on
the top of the network with traditional layers to increase the number of
inception blocks. Using a deeper model than Inception V3, residual
blocks and new architectures or the inception modules, the authors have
achieved a top-5 error rate of 3.08% which make the Inception V4 the
state-of-the-art model over the 2012 challenge.

![51_Inception_ResNet_archi_using_complex_modules](51_Inception_ResNet_archi_using_complex_modules.png)*Inception-ResNet architecture using complex modules. Source: [C. Szegedy and al (2016)](http://arxiv.org/abs/1602.07261)*

To this day, new blocks to improve performances and speed up training
are still invented. For example the “Squeeze-and-Excitation” module
deeply explained by [J. Hu (2017)](https://arxiv.org/abs/1709.01507) uses multiple fully-connected layers and an
architecture inspired from the inception module and the residual block.
It reduces the number of parameters, the computational cost, improve
parallelization using GPU and it has won the 2017 ImageNet challenge
with a top-5 error rate of 2.25%.

![61_SE-ResNet_module](61_SE-ResNet_module.png)*SE-ResNet module. Source: [J. Hu (2017)](https://arxiv.org/abs/1709.01507)*

## Conclusion

This short description of the state-of-the-art on the classification
task of the ImageNet challenge is not exhaustive. It had for purpose to
describe the recent history of deep learning and its expansion for the
common task of classification. The most famous and mostly used
architectures have been quoted but this recent research field is
extremely active.

State-of-the-art models are able to detect an object for a given image,
however the computational cost is still important for training.
Moreover, the size of these models is counted in hundreds of megabytes
even if new architectures have optimize the size while decreasing the
error rate (VGG16 v.s. Inception models). It also means that the
inference has a important computational cost because of the high number
of operations to infer for a given input. This issues constitute a real
subject for embedded deep learning models with Internet of Things
devices (IoT) or mobile devices. Optimization of architectures, weights
storage and computational cost in inference constitute an active field
of research in order to use the power of artificial intelligence in our
everyday objects used.


[^1]: Synonym sets

[^2]: In [C. Szegedy and al (2016)](http://arxiv.org/abs/1602.07261) is developed a pure Inception V4 without residual block and an Inception-ResNet V2 model which uses inception modules and residual blocks. What we call Inception V4 here is the actual Inception-ResNet V2 providing the best performances.
