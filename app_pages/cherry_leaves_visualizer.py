import os
import random
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import streamlit as st


def cherry_leaves_visualizer():
    st.title("Cherry Leaves Visualizer")
    st.info(
        "Business Requirement 1 - Conduct study to visually differentiate "
        "cherry leaf that is healthy from one that contains powder."
    )

    version = 'v1'

    if st.checkbox(
        "Mean and Standard Deviation for Healthy and Mildew Powdery image"
    ):
        st.info(
            "In general, mean and standard deviation represent "
            "average and variability."
        )

        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")
        avg_powdery = plt.imread(
            f"outputs/{version}/avg_var_powdery_mildew.png"
        )

        st.warning(
            "* Output Analysis: The patterns of average and variability "
            "images do not help to intuitively differentiate one from "
            "another. However, both labels display a small difference in "
            "the colour pigment of the average images."
        )

        st.image(
            avg_healthy,
            caption='Healthy Cherry Leaf - Average and Variability'
        )
        st.image(
            avg_powdery,
            caption='Mildew Powdery Cherry Leaf - Average and Variability'
        )
        st.write("---")

    if st.checkbox(
        "Differences between average healthy and average mildew powdery cells"
    ):
        diff_between_avgs = plt.imread(
            f"outputs/{version}/difference_between_averages.png"
        )

        st.warning(
            "* We notice this study didn't show patterns where we could "
            "intuitively differentiate one from another. There is very "
            "little visual difference between the average healthy and "
            "powdery_mildew images. The abs(avg_healthy - avg_mildew) "
            "difference is numerically small — likely in the range "
            "of 0–10 out of 255 per pixel."
        )
        st.image(diff_between_avgs, caption='Difference between avg images')

    if st.checkbox("Image Montage"):
        st.write("* Click 'Create Montage' button to refresh *")
        my_data_dir = 'input/dataset/cherry-leaves'
        validation_dir = os.path.join(my_data_dir, 'validation')
        labels = os.listdir(validation_dir)

        label_to_display = st.selectbox(
            label="Select label",
            options=labels,
            index=0
        )

        if st.button("Create Montage"):
            image_montage(
                dir_path=validation_dir,
                label_to_display=label_to_display,
                nrows=8,
                ncols=3,
                figsize=(10, 25)
            )
        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

    if label_to_display in labels:
        images_list = os.listdir(os.path.join(dir_path, label_to_display))

        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.warning(
                f"Decrease nrows or ncols to create your montage.\n"
                f"There are {len(images_list)} images in your subset, "
                f"but you requested {nrows * ncols} spaces."
            )
            return

        list_rows = range(nrows)
        list_cols = range(ncols)
        plot_idx = list(itertools.product(list_rows, list_cols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        plotted = 0

        for x in range(nrows * ncols):
            try:
                img_path = os.path.join(
                    dir_path, label_to_display, img_idx[x]
                )
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)

                ax = axes[plot_idx[plotted][0], plot_idx[plotted][1]]
                ax.imshow(img_np)
                ax.set_title(
                    f"Width {img_np.shape[1]}px x Height {img_np.shape[0]}px"
                )
                ax.set_xticks([])
                ax.set_yticks([])

                plotted += 1

            except (UnidentifiedImageError, OSError):
                st.warning(f"Skipping unreadable image: {img_idx[x]}")
                continue

            if plotted >= nrows * ncols:
                break

        plt.tight_layout()
        st.pyplot(fig=fig)

    else:
        st.error(f"The label '{label_to_display}' does not exist.")
        st.info(f"Available options: {labels}")


if __name__ == "__main__":
    cherry_leaves_visualizer()