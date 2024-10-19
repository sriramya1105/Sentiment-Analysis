
import streamlit as st
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image, ImageDraw  # Import Pillow for image processing
from nltk.corpus import stopwords
import nltk


# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set up stopwords
stop_words = set(stopwords.words('english'))
print(stopwords.words('english'))


# Function to create a round image
def make_round_image(image_path):
    img = Image.open(image_path).convert("RGBA")
    size = min(img.size)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)
    img.putalpha(mask)
    return img

# --- Sentiment Analysis Creator Section in Sidebar ---
st.sidebar.header("Sentiment Analysis Creator")

# Display creator's photo
image_path = 'Assets/profile_pic.jpg'  # Replace with the actual path to your photo
rounded_image = make_round_image(image_path)
st.sidebar.image(rounded_image, caption="Ramya.c", width=150)

# Display creator's name and new description
st.sidebar.subheader("Ramya")
st.sidebar.write("""
Hi, I'm Ramya, and I'm excited to introduce the Sentiment Analysis feature of this application. 
I am passionate about harnessing the power of AI to understand emotions in text. 
This feature analyzes your input and categorizes it into various emotional labels, such as joy, sadness, anger, and more. 
Enjoy exploring the feelings behind the words!
""")

# Add LinkedIn, GitHub, and email links in the sidebar
st.sidebar.markdown("""
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/k-sri-ramya)  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/sriramya1105)  
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:sriramyak684@gmail.com)
""")

# User Guide Section
st.write("---")
st.header("User Guide")
st.write("""
1. **Input Text**: Enter the text you want to analyze in the text area.
2. **Predict Sentiment**: Click the "Predict" button to analyze the sentiment of the input text.
3. **View Results**: The application will display the predicted sentiment label and associated emotional words.
4. **Clear Input**: If you want to analyze a different text, use the "Clear Input" button to reset the text area.
""")

# Load the model and vectorizer for sentiment analysis
with open('best_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Set up lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatizing(content):
    content = re.sub(r'http\S+|www\S+|https\S+', '', content, flags=re.MULTILINE)
    content = re.sub(r'@\w+', '', content)
    content = re.sub('[^a-zA-Z]', ' ', content)

    content = content.lower()
    content = content.split()
    content = [lemmatizer.lemmatize(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Function to load and resize images
def load_and_resize_image(image_path, size=(300, 300)):
    img = Image.open(image_path)
    img = img.resize(size)
    return img

# Mapping of numerical labels to emotional labels
label_mapping = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

# Lists of words associated with each emotion
emotional_words = {
    "Sadness": ["sad", "sorrow", "depressed", "downcast", "mournful", "unhappy", "grief", "melancholy"],
    "Joy": ["happy", "joyful", "cheerful", "elated", "delighted", "content", "blissful"],
    "Love": ["affection", "adoration", "fondness", "devotion", "passion", "romance"],
    "Anger": ["mad", "furious", "irritated", "enraged", "frustrated", "annoyed"],
    "Fear": ["afraid", "scared", "terrified", "frightened", "anxious", "worried"],
    "Surprise": ["amazed", "astonished", "shocked", "astounded", "stunned", "startled"]
}

# Streamlit UI for sentiment analysis
st.write("---")
st.header("Sentiment Analysis")
st.write("Enter the text to analyze its sentiment:")

# Text input
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Text area
input_text = st.text_area("Input Text", value=st.session_state.input_text, height=150)

# Button to trigger prediction
if st.button("Predict"):
    if input_text:
        processed_text = lemmatizing(input_text)
        text_vector = vectorizer.transform([processed_text])
        predicted_label = best_model.predict(text_vector)

        # Display the predicted label
        result_label = label_mapping[predicted_label[0]]
        st.success(f"Predicted Label: {result_label}")

        # Display associated words and an image based on the predicted label
        if result_label in emotional_words:
            st.write(f"Words associated with {result_label}:")
            st.write(", ".join(emotional_words[result_label]))

            # Image mapping for each emotion (use appropriate paths)
            image_mapping = {
                "Sadness": "Assets/sad.webp",
                "Joy": "Assets/joy.jpg",
                "Love": "Assets/love.png",
                "Anger": "Assets/anger.png",
                "Fear": "Assets/scared.png",
                "Surprise": "Assets/surprise.jpg"
            }

            img = load_and_resize_image(image_mapping[result_label], size=(300, 300))
            st.image(img, caption=result_label.capitalize())
    else:
        st.error("Please enter some text to analyze.")

# Button to clear input
if st.button("Clear Input"):
    st.session_state.input_text = ""  # Clear the input text in session state
