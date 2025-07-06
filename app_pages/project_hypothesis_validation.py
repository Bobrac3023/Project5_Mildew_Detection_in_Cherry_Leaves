import streamlit as st


def project_hypothesis_validation():
    st.title("Project Summary: Mildew Detection in Cherry Leaves")

    st.markdown("## Project Hypothesis")
    st.write(
        """
        The central hypothesis of this project is:

        > Powdery mildew infection in cherry leaves can be visually and
        computationally detected using a machine learning model trained on
        leaf images.

        This hypothesis was driven by the client’s primary goal:
        preventing the supply of compromised agricultural produce to the
        market.

        ### Business Requirements:
        - Visual Differentiation: Determine whether healthy and infected
          leaves can be distinguished by visual characteristics.
        - Predictive Modeling: Develop a model that accurately classifies
          cherry leaves as healthy or infected.
        - User-Friendly Dashboard: Provide technical and non-technical
          insights through an interactive interface.
        """
    )

    st.markdown("## Validation Approach")
    st.write(
        """
        To test the hypothesis, we designed a robust ML pipeline using CNNs.
        Below are the key steps:

        - Data Preparation:
            - Collected image data categorized into healthy and powdery
              mildew classes.
            - Cleaned, resized, and organized the dataset into train, test,
              and validation subsets.
            - Applied image augmentation (rotation, flipping, zoom, etc.) to
              improve generalization.

        - Model Design & Training:
            - Built a Convolutional Neural Network using Keras with:
                - Conv2D layers for feature extraction.
                - Dropout to reduce overfitting.
                - Dense layers for classification.
            - Used binary cross-entropy loss and Adam optimizer.
            - Included early stopping to prevent overfitting.

        - Model Evaluation:
            - Evaluated model accuracy and loss on unseen test data.
            - Monitored performance via learning curves (accuracy/loss).
        """
    )

    st.markdown("## Findings")

    st.subheader("Cherry Leaves Visualizer (Requirement 1)")
    st.write(

            """
        - **Statistical Averages**: The average pixel composition of healthy
        vs. mildew-affected leaves was computed and visualized.
        - **Standard Deviation (Variability)**: Standard deviation
        images highlight greater pixel variability in mildew-infected leaves.
        - **Difference Image**: A pixel-wise difference plot quantitatively
        confirmed visible disparities.
        - **Image Montage**: Random montages of pre-labeled images further
        demonstrated consistent mildew characteristics.
        """
    )

    st.subheader("Mildew Detection Model (Requirement 2)")
    st.write(
        """
        - CNN model achieved 99.2% accuracy on the test set.
        - Loss remained low on validation, confirming good generalization.
        - Prediction performance supports real-time, scalable mildew
          detection.
        """
    )

    st.markdown("## Conclusion")
    st.write(
        """
        The project successfully validates the initial hypothesis:

        - Visual analysis confirms that mildew-infected leaves can be
          distinguished.
        - ML model performance demonstrates strong predictive power.
        - The client’s goal of identifying infected leaves before
          distribution is technically achievable.

        This solution can now be scaled or replicated for other crops with
        similar challenges.
        """
    )

    st.markdown("## Next Steps")
    st.write(
        """
        - Dataset Expansion: Improve model robustness by collecting more
          diverse image samples.
        - Deployment: Wrap the trained model into an API or integrate into a
          farm monitoring system.
        - Hardware Optimization: Train on GPU/TPU for faster results with
          larger datasets.
        - Replicability: Extend this solution to detect other crop diseases
          using similar techniques.
        """
    )


if __name__ == "__main__":
    project_hypothesis_validation()