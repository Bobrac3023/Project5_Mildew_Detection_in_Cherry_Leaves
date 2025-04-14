import streamlit as st

def project_hypothesis_validation():
    st.title("Project Summary: Mildew Detection in Cherry Leaves")

    st.write("""
    ## Project Hypothesis

1. The goal of the client was to make sure that they do not supply the market with a product of compromised quality 
2. During our business assessment phase we understood that using conventional data analysis, it was possible to conduct a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
3. The client had two clear business requirements.
    -   Conduct a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
    -   Predict if a cherry leaf is healthy or contains powdery mildew.
4. The client wanted a dashboard that provides both a non techncial and techncial output


    ## Approach for Validation

1. The machine learning piepline is a sequence of operations that are performed when training a machine learning model. 

-   We completed the following tasks.
    -   Data Collection:  
    -   Data Cleaning or Correcting
    -   Feature Engineering ( We mention this here as there ia an overlap of some tasks between feature engineering and Data Cleaning)
    -   Data Augmentation- Convolution networks struggle to find patterns when the dataset is very limited.
    -   We then split the data in train, test and validation sets.
    -   We trained the data, test the output ad then validated the same 
    -   Convolutional neural network (CNN) model are more modern but computationa; heavy update to Artificaial Neural Networks. As our dataset was images , this was a ntural choice.
    -   Tensor flow a popular Python package using the Sequential Model function to model Neural Networks using different layers was deployed.
    -   Due to its effectiveness and syntax simplicity, another neural network library, known as Keras, was adopted as the interface for TensorFlow from version 2.0.
    -   A Dropout layer is a regularization layer and is used to reduce the chance of **overfitting** the neural network.
    -   With machine learning models we want to analyze the performance of the model over a test set of data that the ML model has not seen at the
        time of training. This performance analysis is called the generalization of the model.
    -   If we get the desired generalized performance, we take these models further for the deployment,otherwise we go for the optimization process

             
    ### Findings 
    1. The findings are captured in the out put folder and also demosntrated in the followwing tabs of the dashboard
    -   Cherry_leaves_visualizer - The three distinct requirements from Requirement 1 are captured here 
        -   Average images and variability images for each class (healthy or powdery mildew),
        -   The differences between average healthy and average powdery mildew cherry leaves,
        -   An image montage for each class.
    2. Mildew_powdery_detection - This tab captures the second business requirement of the cleint in a visual manner
        -   An ML system that is capable of predicting whether a cherry leaf is healthy or contains powdery mildew. 

    ### Visual Differentiation Study
    1. The visual differentiation study revealed significant visual differences between healthy and powdery mildew-infected leaves. 
    2. Average images and variability analyses pointed towards distinctive color and texture patterns 
    3. The powdery mildew is clearly visibilie in innfected images.

    ### Model Training and Evaluation
    1. The CNN model trained on these visual markers achieved an very high level of accuracy with 100% correct prediction of a healthy and infected leaf.
    2. This outcome strongly supports our hypothesis that powdery mildew infection in cherry leaves can be detected using a ML system
    3. The business goal to help the client prevent supply of infected product to the market can be acheived. 
    
    ### Conclusion
    1. What does the client consider as a successful project outcome?
    -   A study showing how to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
    -   The capability to predict if a cherry leaf is healthy or contains powdery mildew.

    The results from both the visual study and the machine learning model validation confirm our hypothesis.     """)

    st.write("""
    ## Next Steps
           
    1. A typical workflow used for supervised learning is: 
        -   Split the dataset into train and test set
        -   Fit the model (either using a pipeline or not)
        -   Evaluate your model. 
    2.  If performance is not good,revisit the process, 
        -   start from data collection
        -   Conduct EDA (Exploratory Data Analysis) etc.
    3. Convolution networks struggle to find patterns when the dataset is very limited, A larger dataset and exploring more sophisticated image processing and machine learning techniques with a faster GPU should help improve the performance of the model
    """)

if __name__ == "__main__":
    project_hypothesis_validation()