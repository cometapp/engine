# Article 3: multiclassification

To define: Pascal VOC, COCO, mAP score, IoU

## Region-based Convolutional Network (R-CNN)

The selective search method developed by [J.R.R. Uijlings and al. (2012)](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf) is an alternative to exhausting seach in an image to capture object location. It initializes small regions in an image and merges them with a hierarchical grouping so that the final group is a box on the entire image. The detected regions are merged according to variety of color spaces and similarity metrics. The output is a few number of region proposals which could contain an object.

![11_selective_search_ex](11_selective_search_ex.PNG)*Selective Search application, top: visualisation of the segmention results of the algorithm, down: visualisation of the region proposals of the algorithm. Source: [J.R.R. Uijlings and al. (2012)](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)*

The Region-based Convolutional Network (R-CNN) model developed by [R. Girshick and al. (2014)](http://islab.ulsan.ac.kr/files/announcement/513/rcnn_pami.pdf) combines the selective search method to detect region proposals and deep learning to find out the object in these regions. 
Each region proposal is resized to match the input of a CNN and it extracts a 4096-dimension vector of features. The features vector is feeded into multiple classifiers to produce probabilities to bellong to each class. Each one of these classes has a SVM classifier trained to infer a probability to detect this object for a given vector of features. This vector also feeds a linear regressor to adapt the shapes of the bounding boxe for a region proposal and thus reduce localization errors.

The CNN model described by the authors is trained on the 2012 ImageNet dataset of the original challenge of image classification. It is fine-tuned using the warped region proposal with IoU (MATH?) >= 0.5 with the ground-truth boxes. Two versions are produced, one version is using the 2012 Pascal VOC dataset and the other the 2013 ImageNet dataset with bounding boxes. The SVM classifiers are also trained for each class of each dataset.

The best R-CNNs models have achieved a 62.4% mAP score over the VOC 2012 test dataset (22.0% increase w.r.t. the second best result on the leader board) and a 31.4% mAP score over the 2013 ImageNet dataset (7.1% increase w.r.t. the second best result on the leader board).

![12_R_CNN](12_R_CNN.PNG)*Region-based Convolution Network (R-CNN). (1) It takes an image as input, (2) exacts the region proposals using selective search, (3) computes the features associated to each region and (4) classifies it using SVMs classifiers. Source: [R. Girshick and al. (2016)](http://islab.ulsan.ac.kr/files/announcement/513/rcnn_pami.pdf)*

![13_R_CNN_blog](13_R_CNN_blog.PNG)*Region-based Convolution Network (R-CNN). Each region proposal feeds a CNN to extract a features vector, possible objects are detected using multiple SVM classifiers and a linear regressor modify the coordinates of the bounding box. Source: [J. Xu's Blog](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)*

## Fast Region-based Convolutional Network (Fast R-CNN)

The purpose of the Fast Region-based Convolutional Network (Fast R-CNN) developped by [R. Girshick (2015)](https://arxiv.org/pdf/1504.08083.pdf) is to reduce the time consumption related to the high number of models necessary to analyse all region proposals. The model describes by the author is called "Fast" R-CNN because the network training is 9 times faster then the basique R-CNN and 213 times faster for inference.

A main CNN with multiple convolutional layers is taking the entire image as input instead of using a CNN for each region proposals (R-CNN). Region of Interests (RoIs) are detected on the produced features maps with the selective search method. Formally, the features maps size is reduced using a RoI pooling layer to get valid Region of Interests by fixing heigh and width as hyper-parameters. Each RoI layer produced feeds fully-connected layers[^1] creating a features vector. The vector is used to predict the observed object with a softmax classfier and to adapt bounding box localizations with a linear regressor. 

The best Fast R-CNNs have reached mAp scores of 70.0% for the 2007 Pascal VOC test dataset, 68.8% for the 2010 Pascal VOC test dataset and 68.4% for the 2012 Pascal VOC test dataset.

![22_Fast_R_CNN_blog](22_Fast_R_CNN_blog.PNG)*The entire image feeds a CNN model to detect RoI on the features maps. Each region is separated using a RoI pooling layer and it feeds fully-connected layers. This vector is used by a softmax classifier to detect the object and by a linear regressor to modify the coordinates of the bounding box. Source: [J. Xu's Blog](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)*

[^1]: The entire architecture is inspired from the VGG16 model, thus it has 16 convolutional layers and 3 fully-connected layers.