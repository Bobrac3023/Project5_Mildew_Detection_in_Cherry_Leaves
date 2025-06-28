import streamlit as st


def project_hypothesis_validation():
    st.title("Project Summary: Mildew Detection in Cherry Leaves")

    st.write(
        """
        ## Project Hypothesis

        1. The goal of the client was to make sure that they do not supply
           the market with a product of compromised quality.
        2. During our business assessment phase we understood that using
           conventional data analysis, it was possible to conduct a study to
           visually differentiate a cherry leaf that is healthy from one that
           contains powdery mildew.
        3. The client had two clear business requirements:
            - Conduct a study to visually differentiate a cherry leaf that is
              healthy from one that contains powdery mildew.
            - Predict if a cherry leaf is healthy or contains powdery mildew.
        4. The client wanted a dashboard that provides both a non-technical
           and technical output.

        ## Approach for Validation

        1. The machine learning pipeline is a sequence of operations that are
           performed when training a machine learning model.

        - Tasks completed:
            - Data Collection
            - Data Cleaning or Correcting
            - Feature Engineering (some overlap with data cleaning)
            - Data Augmentation — CNNs struggle with limited datasets.
            - Data splitting into train, test, and validation sets.
            - Training, testing, and validating the model.
            - CNNs are modern but computationally heavy updates to ANNs.
              Since our dataset consisted of images, CNNs were a natural
              choice.
            - TensorFlow (Sequential Model) was used to create neural
              networks with multiple layers.
            - Keras was used as the high-level interface to TensorFlow 2.0.
            - Dropout layers were used to reduce overfitting.
            - Model generalization was measured on unseen test data.
            - If performance met expectations, the model was retained;
              otherwise optimization was performed.

        ### Findings

        1. Outputs are stored in the `output` folder and shown in dashboard
           tabs.

        - **Cherry_leaves_visualizer** – Requirements from Hypothesis 1:
            - Average and variability images for healthy vs. mildew leaves.
            - Differences between average healthy and mildew-infected leaves.
            - Image montages for each class.
        - **Mildew_powdery_detection** – Meets Business Requirement 2:
            - ML model predicts leaf condition (healthy vs. mildew).

        ### Visual Differentiation Study

        1. The study revealed significant differences between healthy and
           mildew-infected leaves.
        2. Color and texture patterns were visually distinctive.
        3. Mildew was clearly visible in infected images.

        ### Model Training and Evaluation

        1. The CNN achieved near-perfect accuracy in classifying healthy vs.
           infected leaves.
        2. This supports our hypothesis that mildew infection is detectable
           via ML.
        3. Business goal of preventing infected product supply is achievable.

        ### Conclusion

        1. What defines project success for the client?
            - A study showing visual differentiation between healthy and
              infected leaves.
            - A model to predict leaf condition (healthy or mildew).

        Both visual analysis and model validation confirm our hypothesis.
        """
    )

    st.write(
        """
        ## Next Steps

        1. Typical workflow for supervised learning:
            - Split dataset into train/test sets.
            - Fit model (with/without pipeline).
            - Evaluate performance.
        2. If performance is poor, revisit:
            - Data collection.
            - Conduct EDA (Exploratory Data Analysis).
        3. CNNs need larger datasets to improve accuracy. Adding more
           high-quality images and using better hardware (e.g., GPU) can
           further enhance model performance.
        """
    )


if __name__ == "__main__":
    project_hypothesis_validation()