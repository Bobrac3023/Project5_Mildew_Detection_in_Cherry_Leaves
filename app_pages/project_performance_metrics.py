import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import joblib


def project_performance_metrics():
    st.title("Model Performance")
    st.write(
        """
        This section outlines the performance of the machine learning model
        trained to classify cherry leaves as either healthy or infected by
        powdery mildew. We'll look at the accuracy and loss during training
        and validation phases, as well as the model's final performance on
        the test set.

        In our plot, the model behaviour is **overfitting**. We can explain
        this because both loss and accuracy plots for training and validation
        data overshoot per epoch, and the validation accuracy does not
        progress with it. As a result, we see a gap between the training and
        validation accuracy lines.

        Overfitting is common in neural networks. It can be reduced by tuning
        hyperparameters, adding dropout layers, or applying early stopping.
        """
    )

    model_outputs_dir = "outputs/v1"

    training_accuracy_path = os.path.join(
        model_outputs_dir, "model_training_acc.png"
    )
    training_loss_path = os.path.join(
        model_outputs_dir, "model_training_losses.png"
    )

    if os.path.exists(training_accuracy_path) and os.path.exists(
        training_loss_path
    ):
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
    st.write(
        """
        After training, the model was evaluated on a separate test
        set to assess its generalization ability. Here are the results:
        """
    )

    test_performance_path = os.path.join(
        model_outputs_dir, "evaluation.pkl"
    )

    if os.path.exists(test_performance_path):
        try:
            evaluation = joblib.load(test_performance_path)
            if isinstance(evaluation, (list, tuple)) and len(evaluation) == 2:
                df = pd.DataFrame(
                    [evaluation], columns=["Loss", "Accuracy"]
                )
                st.dataframe(df)
            else:
                st.write("Unexpected evaluation format:", evaluation)
        except Exception as e:
            st.error(f"Error loading evaluation file: {e}")
    else:
        st.error("Test set performance data is not available.")


if __name__ == "__main__":
    project_performance_metrics()
