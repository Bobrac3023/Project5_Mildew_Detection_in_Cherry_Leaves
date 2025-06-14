import streamlit as st

# Set page config
st.set_page_config(page_title="Mildew Detection in Cherry Leaves", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation Panel")
options = [
    "Executive project summary",
    "Cherry Leaves visualizer",
    "Mildew_powdery detection",
    "Project hypothesis validation",
    "Project performance metrics"
]
selection = st.sidebar.radio("Select radio button below", options)

# Lazy-load only the selected page
if selection == "Executive project summary":
    from app_pages.executive_project_summary import executive_project_summary
    executive_project_summary()

elif selection == "Cherry Leaves visualizer":
    from app_pages.cherry_leaves_visualizer import cherry_leaves_visualizer
    cherry_leaves_visualizer()

elif selection == "Mildew_powdery detection":
    from app_pages.mildew_powdery_detection import mildew_powdery_detection
    mildew_powdery_detection()

elif selection == "Project hypothesis validation":
    from app_pages.project_hypothesis_validation import project_hypothesis_validation
    project_hypothesis_validation()

elif selection == "Project performance metrics":
    from app_pages.project_performance_metrics import project_performance_metrics
    project_performance_metrics()
