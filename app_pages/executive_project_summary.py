import streamlit as st

def executive_project_summary():
    st.title("Project Title: Mildew Detection in Cherry Leaves")

    st.markdown("""
    ## Project Overview
    ### General Information
    Marianne McGuineys, a fictional individual, is the head of IT and Innovation at Farmy & Foods, a company in the 
    agricultural sector that produces and harvests different types of food. Recently, she is facing a challenge where
    their cherry plantations have been presenting powdery mildew, which is a fungal disease that affects a wide range 
    of plants.The cherry plantation crop is one of their finest products in the portfolio and the company is concerned
    about supplying the market with a product of compromised quality.Currently, the process is to manually verify if a 
    given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples 
    of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If it has powdery mildew, 
    the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. 
    The company has thousands of cherry trees located in multiple farms across the country. As a result, this manual 
    process is not scalable due to time spent in the manual process inspection.To save time in this process, the IT team 
    suggested an ML system that is capable of detecting instantly, using a tree leaf image, if it is healthy or has powdery 
    mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, 
    there is a realistic chance to replicate this project to all other crops. The dataset is a collection of cherry leaf 
    images provided by Farmy & Foods, taken from their crops.""")


    st.markdown("""
    ### Project Dataset 
    1. The dataset is sourced from [Kaggle] https://www.kaggle.com/codeinstitute/cherry-leaves.
    2. The dataset contains +4 thousand images taken from the client's crop fields. 
      The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species.
      The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.""")

    st.markdown("""
    ### Business Requirements
    1. The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
    2. The client is interested in predicting if a cherry tree is healthy or contains powdery mildew.""")

    st.markdown("""
    ### Addtional Information.
    - Additonal information about this project can be seen at this readme file 
      https://github.com/Bobrac3023/Project5_Mildew_Detection_in_Cherry_Leaves/blob/main/README.md""")
    
    st.markdown("""
    ### Dataset Cells Visualizer
    - It will answer business requirements one(1) 
       -   Checkbox 1 - Difference between average and variability image
       -   Checkbox 2 - Differences between average parasitised and average uninfected cells
       -   Checkbox 3 - Image Montage
            1. The checkbox provides the user with a selection of two labels - healthy and powdery mildew.
            2. To create a montage of ramdom pre-labelled images select from the two labels
            3. Every time we click the **Create Montage** button, it generatesa new montage of random images """)

    st.markdown("""
    ### Mildew_powdery detection
    -  This page answers business requirement two (2)
        -   The client is interested in predicting if a cherry tree is healthy or contains powdery mildew
        """)

    st.markdown("""
    ### Project Hypothesis and Validation
    Block for each project hypothesis, describe the conclusion and how you validated it.""")

    st.markdown("""
    ### Project Performance Metrics
    """)

   