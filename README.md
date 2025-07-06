# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)


# Project Overview

This project aims to solve a real-world agricultural challenge: identifying cherry leaves infected by powdery mildew. The goal is to replace the manual, time-consuming inspection process with a scalable, image-based machine learning solution.he cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## General Background

### Client details:

- Marianne McGuineys
  - Head of IT and Innovation
  - Farmy & Foods

Marianne McGuineys, a fictional individual, is the head of IT and Innovation at Farmy & Foods, a company in the agricultural sector that produces and harvests different types of food. Recently, she is facing a challenge where their cherry plantations have been presenting powdery mildew, which is a fungal disease that affects a wide range of plants.

### Business Issue:

- The cherry plantation crop is one of their finest products in the portfolio and the company is concerned about supplying the market with a product of compromised quality.
  - Powdery mildew has been affecting the quality of cherry leaves.
  - Manual visual inspection is slow and labor-intensive.
  - Risk of compromised products reaching the market.

### Objective:

- Build an ML system capable of predicting mildew infection from images
- Create a user-friendly dashboard for visualization and prediction.

## Dataset Content

- Source: Kaggle - Cherry Leaves Dataset sourced from kaggle https://www.kaggle.com/codeinstitute/cherry-leaves
- Content: 4,000+ images of healthy and mildew-infected cherry leaves.
- Structure: Pre-sorted into folders (train/, test/, validation/)
 T
## Business Requirements

Farmy & Foods is currently facing a critical challenge in its cherry plantations, which are increasingly affected by powdery mildew, a fungal disease that compromises crop quality. The organization’s existing inspection method is entirely manual and time-intensive—an employee spends approximately 30 minutes per tree collecting and visually analyzing leaf samples to determine if the plant is healthy or infected. If mildew is detected, a fungicide is applied, which takes an additional 1 minute per tree.

With thousands of cherry trees distributed across multiple farms, this process is not scalable, leading to significant operational inefficiencies and risks of delayed or inconsistent identification of infected trees.

To address this, the IT team proposed the implementation of a machine learning (ML) system capable of analyzing cherry leaf images and providing an instant classification of leaf health. The initiative is not only aimed at solving the current problem in cherry crops but also serves as a pilot model that can be extended to other crops facing similar pest or disease-related challenges.

### Business Requirements 1 and 2

The key business requirements for this project are as follows:

- **Business requirement 1**
  - 1 Difference between average and variability image for each class ( healthy and powdery mildew)
  - 2 The differences between average healthy and average powdery mildew cherry leaves
  - 3 An image montage for each class.
- **Business requirement 2** 
  - Deliver an ML system that is capable of predicting whether a cherry leaf is healthy or contains powdery mildew.


### Goal

- The manual process in place today is not scalable due to time spent in inspection.
- To save time in this process, the IT team suggested an ML system that is capable of detecting instantly, using a tree leaf image, if it is healthy or has powdery mildew.

- **Visual Differentiation:**
  - Develop a visual study to determine whether healthy and infected cherry leaves can be reliably distinguished using image-based features such as color, texture, and shape variations
- **Predictive Modeling:**
  - Design and train a robust Convolutional Neural Network (CNN) model that can accurately classify cherry leaf images into ***"Healthy"*** or ***"Powdery Mildew"*** categories. 
  - The model should demonstrate high generalization ability on unseen test data.
- **Dashboard Output:**
  - Create an interactive dashboard that provides both technical metrics (accuracy, loss curves, confusion matrix) and non-technical insights (clear predictions, sample visuals) to support stakeholders with diverse backgrounds in interpreting model results.

### Client Benefit

- The client will not supply the market with a product of compromised quality.

# Business Assessment Questions - Answered 

1. What are the business requirements?
   - The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
   - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.
2. Is there any business requirement that can be answered with conventional data analysis?
   - Yes, we can use conventional data analysis to conduct a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
3. Does the client need a dashboard or an API endpoint?
   - The client needs a dashboard.
4. What does the client consider as a successful project outcome?
   - A study showing how to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
   - Also, the capability to predict if a cherry leaf is healthy or contains powdery mildew.
5. Can you break down the project into Epics and User Stories?
   - Information gathering and data collection.
   - Data visualization, cleaning, and preparation.
   - Model training, optimization and validation.
   - Dashboard planning, designing, and development.
   - Dashboard deployment and release.
6. Ethical or Privacy concerns?
   - The client provided the data under an NDA (non-disclosure agreement), therefore the data should only be shared with professionals that are officially involved in the project.
7. Does the data suggest a particular model?
   - The data suggests a binary classifier, indicating whether a particular cherry leaf is healthy or contains powdery mildew.
8. What are the model's inputs and intended outputs?
   - The input is a cherry leaf image and the output is a prediction of whether the cherry leaf is healthy or contains powdery mildew.
9. What are the criteria for the performance goal of the predictions?
   - We agreed with the client a degree of 97% accuracy.
10. How will the client benefit?
    - The client will not supply the market with a product of compromised quality.

# PROJECT HYPOTHESEIS AND VALIDATION


###  Hypothesis 

- The client’s primary objective is to ensure that no compromised-quality produce is delivered to the market.
- During the initial business assessment, it was determined that traditional visual inspection techniques could help differentiate between healthy and mildew-infected cherry leaves.
- Two clear business requirements were established:
  - Conduct a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
  - Develop a predictive mechanism to classify a cherry leaf as healthy or infected.
- The client further emphasized the need for a user-friendly dashboard that delivers both technical and non-technical outputs for decision-making.

### Approach for Validation

To validate this hypothesis, a structured machine learning pipeline was implemented, distributed across three main Jupyter notebooks:

#### Data Collection and Preparation 

- The dataset used in this project consists of labeled images of cherry leaves, categorized into two classes: Healthy and Powdery Mildew Infected. 
- These images were sourced from the https://www.kaggle.com/datasets/codeinstitute/cherry-leaves and provided by Farmy & Foods.

#### Preprocessing Workflow 

To ensure the dataset was robust for model training and evaluation, the following preprocessing steps were implemented:

- Data Cleaning
  - Checked and filtered out corrupted or unreadable image files.
  - Standardized image formats to RGB and resized all images to 224x224 pixels to ensure compatibility with the CNN model input layer.
- Feature Engineering (Visual)
  - Though explicit feature extraction was handled by the CNN model, visual differences such as color distribution, texture, and shape variations were explored through:
    - Average and variability image visualizations.
    - Pixel-level difference plots between healthy and infected leaves.
- Data Augmentation
  - To increase the effective training size and improve model generalization, the following augmentation techniques were applied during training:
    - Horizontal and vertical flipping
    - Random rotations
    - Zoom and shift transformations
- Dataset Splitting
  - The cleaned and augmented dataset was split as follows:
    - Training Set: Used for model learning.
    - Validation Set: Used during training for tuning and early stopping.
    - Test Set: Used post-training to evaluate model generalization on unseen data.


#### Model Design and Training

- A **Convolutional Neural Network (CNN)** architecture was selected due to its high efficacy in image classification tasks.
- The model was built using the **TensorFlow Keras API**, leveraging:

  - ***Conv2D, MaxPooling, Dropout,*** and ***Dense*** layers.
  - Dropout layers were included to reduce overfitting.
  - Early stopping was configured to halt training when no further improvement was observed.

#### Model Evaluation

- The model’s performance was evaluated on the **test set**—data unseen during training.
- Key metrics such as **accuracy and loss** were tracked across epochs to monitor training behavior.
- Visualization of learning curves revealed trends in model generalization.

### Conclusion

The hypothesis is validated if:

- There are observable visual distinctions between healthy and infected leaves (confirmed via average/difference visualizations).
- The trained model demonstrates high predictive accuracy (above 99% test accuracy).
- The final dashboard effectively communicates both analytical and actionable insights to stakeholders.

# Model Architecture and Learning Approach

## MACHINE LEARNING PIPELINE

- A typical workflow used for supervised learning is: 
  - Split the dataset into train and test set
  - Fit the model (either using a pipeline or not)
  - Evaluate your model. 
- If performance is not good,revisit the process, 
  - start from data collection
  - Conduct EDA (Exploratory Data Analysis) etc.
  - The Machine learning pipeline can be broken down into three sections as can be seen in the image below.These sections are also used when creating the three Jupyter notebooks

![machine_learning_pipeline](Readme.doc/machine_learning_pipeline.png)


## Model Creation

- The client’s objective is to predict whether a cherry leaf is healthy or affected by powdery mildew. 
- This prediction directly supports functionality within Page 3: mildew_powdery_detection of the Streamlit Dashboard application.
- To achieve this, we implemented a deep learning classification model, with the following rationale and components:

## Why Classification and CNN? 

- CNNs are a modern evolution of artificial neural networks, specially designed for extracting spatial hierarchies in images.
- The input data consists of images, which are unstructured data. For such data, Convolutional Neural Networks (CNNs) are the preferred model architecture.

- A typical CNN architecture includes:
  - Convolutional layers to capture low- and high-level features
  - Pooling layers for dimensionality reduction
  - Fully connected (dense) layers for decision making

![cnn_model](Readme.doc/cnn_model.png)


## Deep Learning and Feature Learning

- The "deep" in deep learning refers to the presence of multiple layers of neurons, enabling the network to learn complex patterns.
- In our case, the CNN learns from the dominant visual feature in labeled images: powdery mildew patterns on cherry leaves.
- The learning process is inspired by human error correction, where mistakes are reduced iteratively using:
  - A loss function to measure prediction error
  - An optimizer to adjust weights and minimize the loss over time


### BIAS & BACKPROPOGATION 

- Deep Neural Networks have two properties namely **BIAS** and **BACKPROPOGATION** due to which we do not have to spend a lot of time doing feature engineering for data. 
- These two functions are used in TensorFlow as **OPTIMIZER** and **LOSS FUNCTIONS**.
  - Tensor flow a popular Python package using the Sequential Model function to model Neural Networks using different layers was deployed.
  - Due to its effectiveness and syntax simplicity, another neural network library, known as Keras, was adopted as the interface for TensorFlow from version 2.0.
  - A Dropout layer is a regularization layer and is used to reduce the chance of **overfitting** the neural network.

- TensorFlow Loss and Optimzation 
  
![tensorflow_loss_optimizer](Readme.doc\tensorflow_loss_optimizer.png)

## INFRASTRUCTURE TOOLS AND TECHNOLOGIES

- The model is built using the TensorFlow deep learning library.
- Training and evaluation details—including architecture, compilation parameters, and performance metrics—are documented in the Modelling and Evaluation Jupyter notebook.
- Programming Language - Python 
- Cloud IDE ( for editors and source control )- We use Github and Jupyter.
- Cloud IDE help us in the CRISP-DM process to complete Data colelction, Visualization, Cleaning along with Model training and evolution into a Jupyter Notebook 
- Dashboard : Streamlit
- Cloud Hosting - Heroku or Render
- Kaggle - This is the location for the images dataset provide by the client 
  - [Kaggle] https://www.kaggle.com/codeinstitute/cherry-leaves 
- Python Data Analysis Packages are captured in the Requirements.txt file and imported inside the Jupyter notebooks
  - numpy==1.26.1
  - pandas==2.1.1
  - matplotlib==3.8.0
  - seaborn==0.13.2
  - plotly==5.17.0
  - Pillow==10.0.1
  - streamlit==1.40.2
  - joblib==1.4.2
  - scikit-learn==1.3.1
  - tensorflow-cpu==2.16.1
  - keras>=3.0.0

# CRISP-DM

- CRISP-DM is the Cross Industry  Standard Process for Data Mining. 
- Through this project we have used this standard while building our three Jupyter notebooks.
  - DataCollection Notebook 
  - DataVisualization Notebook
  - Modelling and Evaluation Notebook
- The CRISP-DM model and the different steps of the model are capture in the images below

## CRISP-DM Model

![crisp_dm_1](Readme.doc/crisp_dm_1.png)
   
## Business Understanding
  
![business_understaning_2](Readme.doc/business_understaning_2.png) 

## Data Understanding

![data_understanding_3](Readme.doc/data_understanding_3.png)

## Data Preparation
  
![data_preparation_4](Readme.doc/data_preparation_4.png) 

## Modelling
  
![modelling5](Readme.doc/modelling5.png) 


# JUPYTER NOTEBOOKS

## DATA COLLECTION NOTEBOOK 

- CRISP-DM Methodology 
  - The task undertaken in this notebook corresponds to the Data Understanding phase of the CRISP-DM methodology, essential for ensuring high-quality input into any ML pipeline.


### Objectives

- Primary Goal
  - Fetch data from the Kaggle dataset hosted by Code Institute (https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) to   initiate the machine learning workflow.
- Extraction and Decomprression 
  - Extract the ***zip*** archive containing two classes — ***healthy and powdery_mildew*** — into a structured directory format suitable for supervised image classification.
- Prepare for Modelling 
  - Clean, preprocess, and structure the dataset into standardized subfolders to support training, validation, and testing stages.
  - Implement data integrity and quality assurance through cleaning and filtering steps.
- CRISP-DM Methodology 
  - This task corresponds to the Data Understanding phase of the CRISP-DM methodology, essential for ensuring high-quality input into any ML pipeline.


### Inputs

- Data Source:
  - Kaggle Dataset: https://www.kaggle.com/datasets/codeinstitute/cherry-leaves
  - ZIP file containing two subfolders: healthy/ and powdery_mildew/
- Directory Structure Post-Extraction:
  - input/dataset/cherry-leaves/
    - healthy/
    - powdery_mildew/
- Environment Setup:
  - Libraries installed via requirements.txt
  - Python dependencies include TensorFlow, Keras, Scikit-learn, Matplotlib, Seaborn, Streamlit, and others.


### Outputs

- The output will stored in the output/dataset folder and pushed to the gitpod repo.  
- Directory After Cleaning & Splitting:
  - input/dataset/cherry-leaves/
    - train/
      - healthy/
      - powdery_mildew/
    - validation/
      - healthy/
      - powdery_mildew/
    - test/
      - healthy/
      - powdery_mildew/
- Output Location:
  - Cleaned and split dataset saved under /input/dataset/cherry-leaves/
- Ready for model training, validation, and final evaluation


### Additional Comments

- Data quality and structure significantly impact model performance. 
- Even if the data is client-provided, validation and cleanup are non-negotiable steps.
- This notebook is part of a modular ML pipeline and precedes:
  - DataVisualization.ipynb (Data Augmentation and EDA)
  - ModellingAndEvaluation.ipynb (Model training, evaluation, and tuning)
- NOTE: CRISP-DM encourages iterative backtracking — poor model performance later may warrant returning to this notebook to enhance data preparation.


## DATA VISUALIZATION NOTEBOOK 

### Objectives

- This notebook addresses Business Requirement 1 defined by the client, focusing on visually distinguishing healthy cherry leaves from those affected by powdery mildew.
- The key deliverables aligned with the Streamlit dashboard ***(Page 2: Cherry Leaves Visualizer)*** include:
  - **Average and Variability Images (Checkbox 1)**:
    - Average images represent the mean pixel values across all samples in each class (healthy or infected).
    - Variability is shown through standard deviation, highlighting intra-class consistency or noise.
    - This visual summary helps the user compare both classes side-by-side using statistical representations.
  - **Differences Between Class Averages (Checkbox 2)**:
    - A third image is presented showing the pixel-wise differences between average healthy and average infected leaves.
    - This helps the user understand subtle variations between the two categories, even if they are not immediately obvious to the human eye.
    - All three visuals—Healthy Average, Infected Average, and Difference Image—are displayed side by side.
  - **Image Montage per Class (Checkbox 3)**:
    - Users can select a label ("Healthy" or "Powdery Mildew") via a dropdown menu.
    - Upon clicking the Create Montage button, a new set of randomly sampled, pre-labeled images is displayed in a montage format.
    - Each click generates a different combination, offering visual intuition on real-world variability within the class.
- This visual analysis supports the client’s goal of assessing whether healthy and infected leaves can be reliably differentiated through human inspection or automated modeling.
  

### Inputs

- The input for this notebook comes from the training, validation, and test datasets that were created in the Data Collection Notebook, located in the following directory structure:
  - Train Dataset - input/dataset/cherry-leaves/test
  - Test Dataset - input/dataset/cherry-leaves/train
  - Validate Dataset - input/dataset/cherry-leaves/validation
  

### Outputs

- A pickle file containing the image shape embeddings.
- Plots visualizing the mean and standard deviation (variability) for each class label (healthy and powdery mildew).
- A contrast plot highlighting the differences between average healthy and mildew-infected images.
- Functional code blocks that fulfill ***Business Requirement 1***, including support for generating image montages within the Streamlit dashboard.

### Importance of this notebook

- This exercise is essential for visually differentiating images between the two classes: **healthy and powdery mildew**.
- In image-based datasets, data visualization is typically limited to image montages that allow us to observe and compare patterns across pre-labeled images.
- By analyzing the statistical differences in mean and variability of each class, we gain insights into the quality and separability of the data — ***a critical step before training any machine learning model***.


## MODELLING AND EVALUATION NOTEBOOK 

### Objectives

- This note book helps meet the clients ***business requirements 2*** as listed below  
- The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.


### Inputs

- The following inputs are used, sourced from the Data Collection notebook:

  - Train Dataset: /input/dataset/cherry-leaves/train

  - Validation Dataset: /input/dataset/cherry-leaves/validation

  - Test Dataset: /input/dataset/cherry-leaves/test

  - Image Shape Embedding: /outputs/v1/image_shape.pkl

  - Label Classes: ['healthy', 'powdery_mildew']
  

### Outputs

- The notebook generates the following outputs, saved in the outputs/v1/ directory:

- class_indices.pkl: For prediction inference in deployed apps.
- mildew_detection_model.h5: Saved trained model.
- evaluation.pkl: Pickled test loss and accuracy.

- Training Performance Visuals:
  - model_training_losses.png
  - model_training_acc.png

- Image Analytics:
  - Average and variability plots per class (avg_var_*.png)
  - Difference between class averages (difference_between_averages.png)

- Predictions on unseen images using saved model.



### Importance of this notebook

- Image Augmentation enhances the training dataset by artificially generating new image variations (in memory) using transformations such as:
  - Random rotations
  - Width/height shifting
  - Shearing
  - Flipping
  - Zooming and rescaling
  - These augmentations expose the model to a wider range of image conditions, improving its ability to generalize and reducing the risk of overfitting.
- Trains a robust CNN-based model, capable of accurately classifying healthy vs. mildew-affected cherry leaves.
- Enables real-time prediction on new, unseen images — a core requirement of the Streamlit dashboard interface.
- Stores essential artifacts for downstream integration:
  - Trained model (.h5)
  - Class mappings (class_indices.pkl)
  - Evaluation metrics (evaluation.pkl)
- Reinforces deployment readiness by:
  - Applying early stopping to prevent overfitting
  - Capturing model performance through accuracy/loss curves
  - Aligning outputs to specific dashboard features



#### Data Augmentation Image
  
![data_augmentation](Readme.doc/data_augmentation.png)

#### Choice of the Algorithm

![algorithm_selection](Readme.doc/algorithm_selection.png)

#### Overfitting Model 
  
![overfitting_model](Readme.doc/overfitting_model.png)

# STREAMLIT APP

## Dashboard Design (Streamlit App User Interface)

- The Streamlit Dashboard is delivered via **Five Distinct app** files. 
- Together, these files form a complete, modular, and interactive machine learning dashboard for the detection of powdery mildew in cherry leaves.
- Below is a comprehensive explanation of each file:
  - how it works individually, 
  - how they complement each other as a cohesive project

## Page 1: Executive_Project Summary

- The Execuive Project Summary is delivered via the ***executive_project_summary*** file

#### Purpose:

- Business context, goals, and scope

#### What it does (Deliverables)

- Introduces the fictional stakeholder : (Marianne McGuineys) and company (Farmy & Foods).
- Describes the problem : manual mildew detection is time-consuming and not scalable.
- Lays out why an ML solution is appropriate.
- Defines two business requirements:
  - Visual differentiation between healthy and mildew-infected leaves.
  - Predictive classification using ML.
- Points to the Kaggle dataset used.
- Summarizes dashboard sections like Cherry Leaves Visualizer, Mildew Detection, etc.

### How it complements others

- Sets the foundation for the hypothesis validation.
- Tells the business story that is answered with the technical notebooks.
- Describes what users will see in other modules (e.g., checkboxes, model output).


## Page 2: Cherry_leaves_Visualizer

- The page is is delivered via the ***cherry_leaves_visualizer*** file

### Purpose:

- Visual study **(addresses Business Requirement 1)**

### What it does (Deliverables)

- Provides an **interactive UI** to explore the dataset visually:
  - Shows **average and standard** deviation images.
  - Displays **pixel-wise differences** between average healthy and mildew-infected leaves.
  - Allows users to generate image montages for visual inspection by class.
- Uses ***matplotlib, seaborn, and PIL*** for image processing.
- Fully interactive using Streamlit checkboxes, selectbox, and image(). 

### How it complements others.

- Provides the visual evidence to support the hypothesis that mildew is visually identifiable.
- Directly supports the **first business requirement** (visual differentiation).
- The ***"Findings"*** section in project_hypothesis_validation refers to this output.
- Data used here is the same dataset used in training the ML model.


## Page 3: Mildew_Powdery_Detection

- The page is is delivered via the ***mildew_powdery_detection*** file

### Purpose: 

- Real-time prediction interface **(addresses Business Requirement 2)**

### What it does

- Loads a trained CNN model **(mildew_detection_model.h5)** from disk.
- Accepts user-uploaded images (.jpg, .jpeg, .png).
- Predicts if the image is Healthy or Powdery Mildew using the model.
- Displays:
- Uploaded image.
- Predicted class and confidence.
- Table of results.
- CSV download of results.
- Uses TensorFlow, PIL, NumPy, and Streamlit components.

### How it complements others

- Implements Business Requirement 2.
- Provides a live demonstration of the model's predictive power.
- Prediction functionality validates the claim in project_hypothesis_validation that mildew is "computationally" detectable.
- Connects user interaction to the model's training results seen in project_performance_metrics.

## Page 4: Project_Hypothesis_Validation

- The page is is delivered via the ***project_hypothesis_validation*** file

### Purpose: 

- Summarize and validate the project hypothesis

### What it does

- Defines the core hypothesis: mildew is detectable both visually and via ML.
- Explains:
  - Business requirements.
  - ML pipeline used (data prep, augmentation, CNN design, evaluation).
  - Visual study outcomes.
  - Model training and test accuracy.
- Includes a clear conclusion and next steps:
  - Scaling.
  - Deployment.
  - Generalizability to other crops.

### How it complements others

- Connects the visual output from cherry_leaves_visualizer.py and the model results from mildew_powdery_detection.py.
- Ties all technical and business findings together to validate project success.
- Serves as the bridge between executive overview and technical proof.

## Page 5: Project_Performance_Metrics

- The page is is delivered via the ***project_performance_metrics*** file

### Purpose:

- Visualize model learning and evaluation

### What it does

- Loads model training curves (accuracy and loss PNGs).
- Reads saved evaluation metrics (evaluation.pkl).
- Displays:
  - Training vs. validation accuracy/loss.
  - Test accuracy/loss as a DataFrame.
- Flags potential overfitting.
- Explains model generalization performance.

### How it complements others

- Reinforces the findings stated in project_hypothesis_validation.py.
- Helps non-technical users visualize what “good model performance” looks like.
- Supports trust in the predictions shown in mildew_powdery_detection.py.

# STREAMLIT APP DASHBOARD - SCREENSHOTS 

- Screenshots of the five pages created on the Streamlit App Dashboard for the client as per their **requirement 1 and 2**

## PAGE 1

- Executive Summary - Business context, goals, and scope

![page1_executive_project_summary](Readme.doc/page1_executive_project_summary.png)


## PAGE 2

- Cherry_leaves_Visualizer - Visual study **(addresses Business Requirement 1)**
  - **Business requirement 1**
    - Difference between average and variability image for each class ( healthy and powdery mildew)
    - The differences between average healthy and average powdery mildew cherry leaves
    - An image montage for each class.

### The user is presented with three tabs on page 2

![page2_three_tabs.png](Readme.doc/page2_three_tabs.png)

### Mean and Standard deviation for Healthy and Mildew Powdery Image - 1 and 2 

#### IMAGE 1

![page2_mean_standadrd_deviation1.png](Readme.doc/page2_mean_standadrd_deviation1.png)

#### IMAGE 2

![page2_mean_standard_deviation2.png](Readme.doc/page2_mean_standard_deviation2.png)

### Difference between average healthy and average mildeww powdery cells

![page2_difference_average_images.png](Readme.doc/page2_difference_average_images.png)


### Image Montage 


#### IMAGE 1 - User selection

- Here the user is asked to click on the "Create montage" button to create a montaage for ***healthy*** or ***powdery mildew leaves*** by selecting the ***Dropdown Label*** option


![page2_click_create_montage.png](Readme.doc/page2_click_create_montage.png)

#### IMAGE 2 - Healthy Montage


![page2_healthy_montage.png](Readme.doc/page2_healthy_montage.png)

#### IMAGE 3 - Powdery Mildew Montage


![page2_powdery_mildew_montage.png](Readme.doc/page2_powdery_mildew_montage.png)


## PAGE 3

- Mildew Powdery Detection - Real-time prediction interface **(addresses Business Requirement 2)**

- **Business requirement 2** 
  - Deliver an ML system that is capable of predicting whether a cherry leaf is healthy or contains powdery mildew.

### IMAGE 1 - User Selection

- Here the user is presented with a screen to drag and drop cherry leaves images for powdery mildew detection.
- The user is also provided with a link to down the images from the Kaggle dataset 

![page3_client_selection.png](Readme.doc/page3_client_selection.png)

![page3_popup_to_upload.png](Readme.doc/page3_popup_to_upload.png)

### IMAGE 2 - Heathy Leaf

![page3_healthy_leaf.png](Readme.doc/page3_healthy_leaf.png)

- The user can down the predection score in a CSV File 

![page3_healthy_leaft_prediction.png](Readme.doc/page3_healthy_leaft_prediction.png)



### IMAGE 3 - Leaf with Mildew Powdery Fungal Infection

- NOTE: The user can upload muitple files as seen below 

![page3_multiple_files.png](Readme.doc/page3_multiple_files.png)

![page3_powdery_mildew_leaf.png](Readme.doc/page3_powdery_mildew_leaf.png)

- The user can down the predection score in a CSV File 

![page3_powdery_mildew_prediction.png](Readme.doc/page3_powdery_mildew_prediction.png)


## PAGE 4

- Project Hypothesis Validation -Summarize and validate the project hypothesis
  
![page4.png](Readme.doc/page4.png)

## PAGE 5

- Project Performance Metrics - Visualize model learning and evaluation.

![page5_model_performance.png](Readme.doc/page5_model_performance.png)

![page5_test.png](Readme.doc/page5_test.png)


# UNFIXED BUGS

## Heroku Deployment

- A lot of issues was encountered while working on this project.
  - The Walkthrough project and code is built on python 2.8 and the code institite template had 3.12 as the latest version.
  - While deploying to Heroku, the slug was over 500 MB which prevented from deploying on Heroku with all relevant files and libraries.
  - In order to deploy on Heroku many deletions were made 
    - streamlit==1.40.2
    - Pillow==10.0.1
    - numpy==1.26.1
    - tensorflow-cpu==2.16.1
    - matplotlib==3.8.0
    - pandas==2.1.1
    - seaborn==0.13.2
  - The input directories which include the test, train and validation sets were deleted to reduce the slug file size. 
  - As such the Heroku app deployed did not have the input/validation files which are essential to display features on the streamlit app. 
- On rasining this issue through tutor support, the problem was acknowledged as a "Known Issue"
  - Tutor support suggested moving to a different platform called "Render".
- A screenshot of the issue with Heroku.
  
![heroku_slug_fail](Readme.doc/heroku_slug_fail.png) 

## Render Deployment

- Deployment on Render also presented numerous challenges, as it kept throwing errors for Pandas and other packages.
- An example of the error is pasted below 
  - We used a lower version of Numpy than what was in my original requirements.txt file , deleted Procfile and runtime.txt files and used a Python version of 3..8.12 as outlined in the code institue deployment guide.

- Render Deploy issues 
  
![render_deploy_issue](Readme.doc/render_deploy_issue.png)
  
# PROJECT DEPLOYMENT

- Project deployment link at Render https://project5-mildew-detection-in-cherry-5ijr.onrender.com
- Github Repository https://github.com/Bobrac3023/Project5_Mildew_Detection_in_Cherry_Leaves.git

### Why Render and not Heroku 

- As stated earlier Heroku site has a limitation of 500MB on the slug file. 
- Code Institute suggests using an alternate site called Render.com https://dashboard.render.com/web/new
- Deployment guide for Render https://code-institute-students.github.io/deployment-docs/42-pp5-pa/



# APP FILES VALIDATION 

## Cherry Leaves Visualizer

![cherry_leaves_visualizer.png](Readme.doc/cherry_leaves_visualizer.png)

## Executive Project Summary

![executive_project_validation.png](Readme.doc/executive_project_validation.png)


## Mildew Powdery Detection 

![mildew_powdery_detection_validation.png](Readme.doc/mildew_powdery_detection_validation.png)

## Project Hypothesis Validation 

![project_hypothesis_validation.png](Readme.doc/project_hypothesis_validation.png)

## Project Performance Metrics

![project_performance_metrics.png](Readme.doc/project_performance_metrics.png)


## Credits

- Code Institute Documentation: Definitions and other key concepts related to CNN models were sourced from the learning notebooks
- Model architecture and base code were adapted from Gyan Shashwat’s Walkthrough Project 1 – Malaria Detector.
- Streamlit dashboard structure was inspired by Jordon Fletorides' mildew detection project.
- Dataset provided by Code Institute via Kaggle – Cherry Leaves Dataset.
- Pandas Categorical Documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html
- Keras Image Augmentation Guide: https://pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
- Deployment Guide (Render): https://code-institute-students.github.io/deployment-docs/42-pp5-pa/
- JSON Validation: https://jsonlint.com/
- Python Code Linter: https://pep8ci.herokuapp.com/
- Code Institute – How to Use Jupyter Notebooks: https://github.com/Code-Institute-Solutions/how-to-use-jupyter-notebooks
- Streamlit API Documentation: https://docs.streamlit.io/develop/api-reference
- ChatGPT https://chatgpt.com/?model=gpt-4o

### Media

- All Images in this Readme file are sourced from the lessons covered in the Predictive analytics course at Code Institute.

## Acknowledgements 


- Gyan Shashwat – A significant portion of the model logic and structure was inspired by his detailed walkthrough in Project 1: Malaria Detector. His clear explanations greatly enhanced understanding and implementation.

- Neil and Fernando Doritu – Their engaging and well-structured course modules provided an excellent foundation for understanding CNNs and model training.

- Rohit Sharma (Mentor) – Special thanks to my mentor for his unwavering support, especially for accommodating spontaneous calls after long days. His guidance was instrumental throughout the project.

- Jordon Fletorides – The design and structure of the Streamlit dashboard were influenced by Jordon’s excellent work in his mildew detection project, which served as both reference and inspiration.

- Thanks to ChatGPT by OpenAI for assisting with clarifying concepts, refining technical explanations, and polishing documentation during this project.
