# Article 3: multiclassification

To define: Pascal VOC, COCO, mAP score, IoU

## Region-based CNN (R-CNN)

The selective search method developed by [J.R.R. Uijlings and al. (2012)](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf) is an alternative to exhausting seach in an image to capture object location. It initializes small regions in an image and merges them with a hierarchical grouping so that the final group is a box on the entire image. The detected regions are merged according to variety of color spaces and similarity metrics between proposal regions.

![11_selective_search_ex](11_selective_search_ex.PNG)*Selective Search application, top: visualisation of the segmention results of the algorithm, down: visualisation of the region proposals of the algorithm. Source: [J.R.R. Uijlings and al. (2012)](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)*

The Region-based Convolutional Network (R-CNN) model developed by [R. Girshick and al. (2016)](http://islab.ulsan.ac.kr/files/announcement/513/rcnn_pami.pdf) combines the selective search method to detect region proposals and deep learning to find out the object in these regions. 
Each region proposal is resized to match the input of a CNN and it extracts a 4096-dimension vector of features. The feature vector is feeded into multiple classifiers to produce probabilities to bellong to each class. Each one of these classes has a SVM classifier trained to infer a probability to detect this object for a given vector of features.

The CNN model described by the authors is trained on the 2012 ImageNet dataset of the original challenge of image classification. It is fine-tuned using the warped region proposal with IoU (MATH?) >= 0.5 with the ground-truth boxes. Two versions are produced, one version is using the 2012 Pascal VOC dataset and the other the 2013 ImageNet dataset with bounding boxes. The SVM classifiers are also trained for each class of each dataset.

The best R-CNNs models have achieved a 62.4% mAP score over the VOC 2012 test dataset (22% increase w.r.t. the second best result on the leader board) and a 31.4% mAP score over the 2013 ImageNet dataset (7.1% increase w.r.t. the second best result on the leader board).

![12_R_CNN](12_R_CNN.PNG)*Region-based Convolution Network (R-CNN). (1) It takes an image as input, (2) exacts the region proposals using selective search, (3) computes the features associated to each region and (4) classifies it using SVMs classifiers. Source: [R. Girshick and al. (2016)](http://islab.ulsan.ac.kr/files/announcement/513/rcnn_pami.pdf)*