
import streamlit as st
# load pages scripts
from app_pages.executive_project_summary import executive_project_summary
from app_pages.cherry_leaves_visualizer import cherry_leaves_visualizer
from app_pages.mildew_powdery_detection import mildew_powdery_detection
from app_pages.project_hypothesis_validation import project_hypothesis_validation
from app_pages.project_performance_metrics import project_performance_metrics

# Set page config
st.set_page_config(page_title="Mildew Detection in Cherry Leaves", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation Panel")
options = ["Executive project summary", "Cherry Leaves visualizer", "Mildew_powdery detection", "Project hypothesis validation", "Project performance metrics"]
selection = st.sidebar.radio("Select radio button below", options)

# Page display
if selection == "Executive project summary":
    executive_project_summary()
elif selection == "Cherry Leaves visualizer":
    cherry_leaves_visualizer()
elif selection == "Mildew_powdery detection":
    mildew_powdery_detection()
elif selection == "Project hypothesis validation":
    project_hypothesis_validation()
elif selection == "Project performance metrics":
    project_performance_metrics()

    #app.run()  # Run the app