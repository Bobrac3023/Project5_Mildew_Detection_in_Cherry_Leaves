import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image

def project_performance_metrics():
    st.title("Model Performance")
    st.write("""
    This section outlines the performance of the machine learning model trained to classify cherry leaves as either healthy or infected by powdery mildew. 
    We'll look at the accuracy and loss during training and validation phases, as well as the model's final performance on the test set.
    In our plot, the model behaviour is **overfitting**.
    We can explain this because both loss and accuracy plots for training and validation data overshooting per epoch, and the validation accuracy does not progress with it.
    As a result, we would see a gap between the training and validation accuracy lines.
-   Generally, overfitting is a more common phenomenon in neural networks. We can reduce it by tuning our model hyperparameters
    """)

    # Update base path for model outputs
    model_outputs_dir = "outputs/v1"
    

    # Load and display the training accuracy and loss plots
    training_accuracy_path = os.path.join(model_outputs_dir, "model_training_acc.png")
    training_loss_path = os.path.join(model_outputs_dir, "model_training_losses.png")

    if os.path.exists(training_accuracy_path) and os.path.exists(training_loss_path):
        training_accuracy = Image.open(training_accuracy_path)
        training_loss = Image.open(training_loss_path)

        col1, col2 = st.columns(2)
        with col1:
            st.image(training_accuracy, caption="Training Accuracy")
        with col2:
            st.image(training_loss, caption="Training Loss")
    else:
        st.error("Training accuracy and loss plots are not available.")

    st.write("## Test Set Performance")
    st.write("""
    .After training, the model was evaluated on a separate test set to assess its generalization ability. Here are the results:
    """)

    # Update path for test set performance metrics (assuming it's saved as a .pkl for demonstration)
    test_performance_path = os.path.join(model_outputs_dir, "evaluation.pkl")
    
    if os.path.exists(test_performance_path):
        test_performance_df = pd.read_pickle(test_performance_path)
        st.dataframe(test_performance_df)
    else:
        st.error("Test set performance data is not available.")

if __name__ == "__main__":
    project_performance_metrics()
