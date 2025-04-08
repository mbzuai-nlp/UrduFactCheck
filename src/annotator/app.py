import os
import json
import streamlit as st
from arabic_support import support_arabic_text

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Urdu Fact Checking Annotation Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    "<h3 style='text-align: center;'>Urdu Fact Checking Annotation Dashboard</h3>",
    unsafe_allow_html=True,
)


# Load JSON data with caching
@st.cache_data(show_spinner=False)
def load_json(file):
    return json.load(file)


# File Upload and Data Loading
uploaded_file = st.file_uploader("Choose a JSON file to annotate:", type=["json"])
if not uploaded_file:
    st.info("Please upload a file to get started.")
    st.stop()

# Load the JSON data
data = load_json(uploaded_file)
if isinstance(data, dict):
    data = [data]
elif not isinstance(data, list):
    st.error("Invalid JSON structure. Expecting a list or a dict.")
    st.stop()
if not data:
    st.error("Empty JSON list.")
    st.stop()


# Initialize session state variables
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = []
if "live_annotation" not in st.session_state:
    st.session_state.live_annotation = {
        "question_urdu": "",
        "answer_urdu": "",
    }
# Navigation Buttons: update index and force re-run
col_nav1, col_nav2, col_nav3 = st.columns(3)
with col_nav1:
    if st.button("Previous", use_container_width=True):
        # Reset annotation controls
        st.session_state["question_urdu"] = ""
        st.session_state["answer_urdu"] = ""

        # Update index
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1
with col_nav2:
    if st.button("Next", use_container_width=True):
        # Reset annotation controls
        st.session_state["question_urdu"] = ""
        st.session_state["answer_urdu"] = ""

        # Update index
        if st.session_state.current_index < len(data) - 1:
            st.session_state.current_index += 1
with col_nav3:
    if st.button("Save and Next", use_container_width=True):
        # Save the current annotation
        # Update live annotation with the text box values
        st.session_state.live_annotation["question_urdu"] = st.session_state[
            "question_urdu"
        ]
        st.session_state.live_annotation["answer_urdu"] = st.session_state[
            "answer_urdu"
        ]

        # Save the annotation for the current item
        current_item = data[st.session_state.current_index]
        annotated_item = current_item.copy()
        annotated_item.update(st.session_state.live_annotation)
        st.session_state.annotations.append(annotated_item)

        os.makedirs("tmp", exist_ok=True)
        temp_filename = "tmp/" + uploaded_file.name.replace(".json", "_human_temp.json")
        with open(temp_filename, "w") as f:
            json.dump(st.session_state.annotations, f, indent=4, ensure_ascii=False)

        # Reset annotation controls for the next round
        st.session_state["question_urdu"] = ""
        st.session_state["answer_urdu"] = ""

        # Update index to the next item
        if st.session_state.current_index < len(data) - 1:
            st.session_state.current_index += 1

# Get current item based on index
current_item = data[st.session_state.current_index]

# Display the current question after navigation updates
st.text_area(value=current_item["question"], label="Question", height=68, disabled=True)
st.text_area(value=current_item["answer"], label="Answer", height=68, disabled=True)
st.session_state["question_urdu"] = st.text_area(
    value=current_item["question_urdu"], label="Question Urdu", height=68
)
st.session_state["answer_urdu"] = st.text_area(
    value=current_item["answer_urdu"], label="Answer Urdu", height=68
)

if st.session_state.annotations:
    st.download_button(
        label="Download Annotations",
        data=json.dumps(st.session_state.annotations, indent=4, ensure_ascii=False),
        file_name=uploaded_file.name.replace(".json", "_human.json"),
        mime="application/json",
    )
