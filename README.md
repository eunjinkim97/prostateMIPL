

<h1 align="center">
    <p> Prediction of BCR for Prostate MRI by using deep learning based algorithm </p>
    
### Code for Clinical Study
    - This code is about a study for analyzing prostate cancers using by deep learning and machine learning. 
    - The authors of the study are
            Hye Won Lee 1,# , Eunjin Kim 2,# , Inye Na 2 , Chan Kyo Kim 3,* , Seong Il Seo 1,* , Hyunjin Park 2,4,*
    - Affiliations:
            1 Samsung Medical Center, Department of Urology, Sungkyunkwan University School of Medicine, Seoul, Korea
            2 Department of Electrical and Computer Engineering, Sungkyunkwan University, Suwon, Korea
            3 Department of Radiology and Center for Imaging Science, Samsung Medical Center, Sungkyunkwan University School of Medicine, Seoul, Korea
            4 Center for Neuroscience Imaging Research, Institute for Basic Science, Suwon, Korea
    - This code for the study was written by Eunjin Kim and Inye Na from the Medical Imaging Processing Lab(MIPL) led by Hyunjin Park in 2022 from SKKU.

### Keywords
MRI, Prostate, Cancer, Multi-parametric, BCR, Deep-learning, Radiomics
    
</h1>


## Overview
The code includes three main parts:

1. Data_preprocessing.py
    - Preprocessing of MR images: This includes preprocessing of multi-parametric images (T1, T2, and T1 contrast-enhanced) of prostate MRIs.

2. Train_and_Extract_DL_features.py
    - Deep learning network for BCR prediction: This includes training a deep learning network to predict BCR by taking 3 inputs as MR images mentioned above.
    - Extraction for deep features associated BCR: 960-features are extracted at the end of deep learning networks.

3. Survival_model.m
    - Survival model techniques: This includes fitting a survival model called Cox modeling.


## Requirements
Python-3.2, 
TensorFlow-2.8, 
NumPy, 
Pandas, 
Scikit-learn, 
SimpleITK,
PyDicom


## Code Contributors
Eunjin Kim, 
Inye Na
  
## Acknowledgments
We would like to thank Hyunjin Park and the Medical Imaging Processing Lab for their support and guidance in conducting this study.

## Lincense

/*******************************************************

 Copyright (C) 2022 Eunjin Kim <dmswlskim970606@gmail.com>
 
 This file is a part of Prediction BCR for Prostate cancer.
 
 This project and code can not be copied and/or distributed without the express permission of EJK, skkuej.

 *******************************************************/
