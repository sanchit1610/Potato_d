import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import openai

# Configure the OpenAI API
openai.api_key = "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"  # Replace with your actual API key

# Define a chatbot function using OpenAI API
def chatbot_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the gpt-3.5-turbo model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

# Streamlit app
st.set_page_config(page_title="Potato Disease Detector", layout="centered")
st.title("🥔 Potato Leaf Disease Classifier")
st.write("This app predicts potato leaf diseases using a deep learning model.")

# Chatbot section
st.sidebar.title("🤖 Chatbot")
user_input = st.sidebar.text_input("Ask me anything about potato diseases:")
if user_input:
    response = chatbot_response(user_input)
    st.sidebar.write(f"Bot: {response}")

# Add a tab for the image gallery
tab1, tab2 = st.tabs(["📷 Image Gallery", "🔍 Disease Detector"])

with tab1:
    st.header("📷 Image Gallery")
    st.write("Browse through examples of healthy and diseased potato leaves.")

    # Define the categories and their display names
    categories = {
        "Potato___healthy": "Healthy",
        "Potato___Early_blight": "Early Blight",
        "Potato___Late_blight": "Late Blight"
    }

    # Create a dropdown to select the category
    selected_category = st.selectbox("Select a category:", list(categories.keys()), format_func=lambda x: categories[x])

    # Replace this with your actual path to the PlantVillage folder
    # Example for Windows: plantvillage_dir = r"C:\Users\YourName\Projects\PlantVillage"
    # Example for macOS/Linux: plantvillage_dir = "/Users/YourName/Projects/PlantVillage"
    plantvillage_dir = "/Users/sanchitthakur/Desktop/potato/PlantVillage"  # Replace this line

    # Get the list of images for the selected category
    image_dir = os.path.join(plantvillage_dir, selected_category)

    # Get all image files
    image_paths = glob.glob(os.path.join(image_dir, "*.JPG")) + \
                  glob.glob(os.path.join(image_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(image_dir, "*.png"))

    if not image_paths:
        st.warning(f"No images found for the selected category: {categories[selected_category]}")
    else:
        # Limit to the first 9 images
        image_paths = image_paths[:9]

        # Display the images in a grid
        cols = st.columns(3)  # Create 3 columns for the grid
        for i, img_path in enumerate(image_paths):
            try:
                with cols[i % 3]:  # Alternate between the columns
                    img = Image.open(img_path)
                    st.image(img, caption=categories[selected_category], use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image {img_path}: {str(e)}")

with tab2:
    # Debug log
    

    # Load model
    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model("potato_disease_model.h5")
        return model

    
    model = load_model()
    

    # Define class names
    CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

    # File uploader
    input_mode = st.radio("Select Image Input Method:", ["Upload", "Camera"])

    language = st.selectbox("Select Language", ["English", "Hindi"])

    if input_mode == "Upload":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    elif input_mode == "Camera":
        uploaded_file = st.camera_input("Take a picture")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        st.write("⏳ Predicting...")
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Prediction: **{CLASS_NAMES[class_idx]}**")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("📊 Prediction Confidence")
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, prediction[0], color='skyblue')
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])
        st.pyplot(fig)

    # Define disease_info dictionary
    disease_info = {
        'Early Blight': {
            'English': {
                'description': 'Early blight is a common fungal disease affecting potatoes.',
                'treatment': 'Use fungicides and practice crop rotation.'
            },
            'Hindi': {
                'description': 'आलू को प्रभावित करने वाला एक सामान्य फफूंद जनित रोग है।',
                'treatment': 'फफूंदनाशकों का उपयोग करें और फसल चक्रण का अभ्यास करें।'
            }
        },
        'Late Blight': {
            'English': {
                'description': 'Late blight is a serious fungal disease affecting potatoes.',
                'treatment': 'Use fungicides and remove infected plants.'
            },
            'Hindi': {
                'description': 'आलू को प्रभावित करने वाला एक गंभीर फफूंद जनित रोग है।',
                'treatment': 'फफूंदनाशकों का उपयोग करें और संक्रमित पौधों को हटाएं।'
            }
        },
        'Healthy': {
            'English': {
                'description': 'The plant is healthy.',
                'treatment': 'No treatment needed.'
            },
            'Hindi': {
                'description': 'पौधा स्वस्थ है।',
                'treatment': 'कोई उपचार आवश्यक नहीं है।'
            }
        }
    }

    # Optional: Add description and treatment
    if uploaded_file is not None:
        st.write(disease_info[CLASS_NAMES[class_idx]][language]['description'])
        st.write(disease_info[CLASS_NAMES[class_idx]][language]['treatment'])
