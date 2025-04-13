# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Template Instructions

Welcome,

This is the Code Institute student template for the Cherry Leaves project option in Predictive Analytics. We have preinstalled all of the tools you need to get started. It's perfectly okay to use this template as the basis for your project submissions. Click the `Use this template` button above to get started.

You can safely delete the Template Instructions section of this README.md file and modify the remaining paragraphs for your own project. Please do read the Template Instructions at least once, though! It contains some important information about the IDE and the extensions we use.

## How to use this repo

1. Use this template to create your GitHub project repo

1. In your newly created repo click on the green Code button. 

1. Then, from the Codespaces tab, click Create codespace on main.

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and `pip3 install -r requirements.txt`

1. Open the jupyter_notebooks directory, and click on the notebook you want to open.

1. Click the kernel button and choose Python Environments.

Note that the kernel says Python 3.12.1 as it inherits from the workspace, so it will be Python-3.12.1 as installed by Codespaces. To confirm this, you can use `! python --version` in a notebook code cell.

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you, so do not share it. If you accidentally make it public, then you can create a new one with _Regenerate API Key_.

## Dataset Content

- The dataset is sourced from [Kaggle] https://www.kaggle.com/codeinstitute/cherry-leaves. We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and how to validate?

- List here your project hypothesis(es) and how you envision validating it (them).

## The rationale to map the business requirements to the Data Visualisations and ML tasks

- List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.

## ML Business Case

- In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

## Dashboard Design (Streamlit App User Interface)


### Page 1: Quick Project Summary


#### General Information

Marianne McGuineys, a fictional individual, is the head of IT and Innovation at Farmy & Foods, a company in the agricultural sector that produces and harvests different types of food. Recently, she is facing a challenge where their cherry plantations have been presenting powdery mildew, which is a fungal disease that affects a wide range of plants.

The cherry plantation crop is one of their finest products in the portfolio and the company is concerned about supplying the market with a product of compromised quality.

Currently, the process is to manually verify if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If it has powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located in multiple farms across the country. As a result, this manual process is not scalable due to time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that is capable of detecting instantly, using a tree leaf image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project to all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

#### The business requirements are:

- The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
- The client is interested in predicting if a cherry tree is healthy or contains powdery mildew.

#### Deliverables 

- Deliver a dashboard that meets the above requirements.

### Page 2: Cells Visualizer

* It will answer business requirement 1
   - Checkbox 1 - Difference between average and variability image for each class ( healthy and powdery mildew)
   - Checkbox 2 - The differences between average healthy and average powdery mildew cherry leaves
   - Checkbox 3 - An image montage for each class.

### Page 3: Healthy or Powdery Detector

- Business requirement 2 information - "an ML system that is capable of predicting whether a cherry leaf is healthy or contains powdery mildew."
- A link to download a set of cherry leaf images for live prediction (https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).
- A User Interface with a file uploader widget. The user should have the capacity to upload multiple images. For each image, it will display the image and a prediction statement, indicating if a cherry leaf is healthy or contains powdery mildew and the probability associated with this statement.
- A table with the image name and prediction results, and 
- A download button to download the table.
  

### Page 4: Project Hypothesis and Validation

- Block for each project hypothesis, describe the conclusion and how you validated.
  
### Page 5: ML Performance Metrics

- Label Frequencies for Train, Validation and Test Sets
- Model History - Accuracy and Losses
- Model evaluation result
- List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items, that your dashboard library supports.
- Finally, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project, you were confident you would use a given plot to display an insight, but later, you chose another plot type).

# MACHINE LEARNING PIPELINE 

# DATA COLLECTION NOTEBOOK 

### Objectives

* Fetch data from Kaggle dataset . The datasource for this project has been provided by cCode institute at https://www.kaggle.com/datasets/codeinstitute/cherry-leaves in the form a zip file 
* Extract the Zip file , and prepare it for further Machine Learning analysis 
* Save the file in out/dataset/ folder and push it to the repo 
  

### Inputs

* The input for this notebook is a Kaggle dataset from Code Insittute at https://www.kaggle.com/datasets/codeinstitute/cherry-leaves
* This zip file is saved and extracted at input/datase/cherry_leaves/folder 
* There are Two  file - Healthy and power_mildew

### Outputs

* The output will stored in the output/dataset folder and pushed to the gitpod repo.  

### Additional Comments

* In case you have any additional comments that don't fit in the previous bullets, please state them here. 



# DATA VISUALIZATION NOTEBOOK 

### Objectives

* This note book helps meet the clients business requirements 1 as listed below  
  - Average images and variability images for each class (healthy or powdery mildew). 
    - In general the mean and standard deviation is called avergae and variablity. 
    - This will help us meet the Checkbox 1 of Page 2 on our Steamlit Dashboard App
  - The differences between average healthy and average powdery mildew cherry leaves. 
    - This will help us meet the Checkbox 2 of Page 2 on our Steamlit Dashboard App. 
    - We can see three images - Average healthy, Avergae Powdery and Difference in healthy and powdery
  - An image montage for each class - healthy and Powdery Mildew cherry leaves
    - In the Streamlit Dasboard app under Page 2 the client can select a labeel - Healthy or Powdery Mildew
    - This will allow the client to create a montage of ramdom pro-labelled images from the selected images for the selected label.
    - Every time the client clicks the **Create Montage** button, it generates a new montage of random images 
  

### Inputs

* The input for this notebook from the test, train and vaidation datasets created in the DataColelction notebook under the below directories
  - Train Dataset - /workspaces/Project5_Mildew_Detection_in_Cherry_Leaves/input/dataset/cherry-leaves/train
  - Test Dataset - /workspaces/Project5_Mildew_Detection_in_Cherry_Leaves/input/dataset/cherry-leaves/test
  - Validate Dataset - /workspaces/Project5_Mildew_Detection_in_Cherry_Leaves/input/dataset/cherry-leaves/validation
  

### Outputs

* The output will as below 
  - Image shape embedding pickle file 
  - Mean and Variability of imagess per label plot 
  - Plot to distinguish contrast between parasite-contained and uninfected cell images
  - Generate code that answers business requirement 1 and can be used to build image montage on Streamlit dashboard

### Importance of this notebook

  - This exercise is important to visually differentiate images of one class from another.
  - data visualization for image data is usually limited to creating animage montage to visually differentiate between different pre-labeled images.
  - Understanding the statistical difference between the mean and variability of the images of different classes helps you to anticipate the quality of data for model training.

# MODELLING AND EVALUATION NOTEBOOK 

### Objectives

* This note book helps meet the clients business requirements 2 as listed below  
  - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.


### Inputs

* The input for this notebook from the test, train and vaidation datasets created in the DataColelction notebook under the below directories
  - Train Dataset - /workspaces/Project5_Mildew_Detection_in_Cherry_Leaves/input/dataset/cherry-leaves/train
  - Test Dataset - /workspaces/Project5_Mildew_Detection_in_Cherry_Leaves/input/dataset/cherry-leaves/test
  - Validate Dataset - /workspaces/Project5_Mildew_Detection_in_Cherry_Leaves/input/dataset/cherry-leaves/validation
  - image shape embeddings
  

### Outputs

* The output will as below 
  - Images distribution plot in train, validation, and test set 
  - Image augmentation 
  - Class indices to change prediction inference in labels
  - Machine learning model creation and training
  - Save model
  - Learning curve plot for model performance 
  - Model evaluation on pickle file
  - Prediction on the random image file

### Importance of this notebook

  - Image augmentation increases the training image data by artificially and temporarily creating training images through different processes,
or a combination of multiple processes, such as random rotation, shifts, shear, and flips, etc, in the computer’s short term memory.

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.
