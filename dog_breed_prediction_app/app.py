from dog_description import dog_descriptions

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model (cached to avoid reloading every time)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        'model.h5',
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    return model

model = load_model()

# Preprocess the image
IMG_SIZE = 224

def preprocess_image(image_path, img_size=IMG_SIZE):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[img_size, img_size])
    return image

# Full list of class names
CLASS_NAMES = [
    "affenpinscher", "afghan_hound", "african_hunting_dog", "airedale",
    "american_staffordshire_terrier", "appenzeller", "australian_terrier",
    "basenji", "basset", "beagle", "bedlington_terrier", "bernese_mountain_dog",
    "black-and-tan_coonhound", "blenheim_spaniel", "bloodhound", "bluetick",
    "border_collie", "border_terrier", "borzoi", "boston_bull", "bouvier_des_flandres",
    "boxer", "brabancon_griffon", "briard", "brittany_spaniel", "bull_mastiff",
    "cairn", "cardigan", "chesapeake_bay_retriever", "chihuahua", "chow", "clumber",
    "cocker_spaniel", "collie", "curly-coated_retriever", "dandie_dinmont", "dhole",
    "dingo", "doberman", "english_foxhound", "english_setter", "english_springer",
    "entlebucher", "eskimo_dog", "flat-coated_retriever", "french_bulldog",
    "german_shepherd", "german_short-haired_pointer", "giant_schnauzer", "golden_retriever",
    "gordon_setter", "great_dane", "great_pyrenees", "greater_swiss_mountain_dog",
    "groenendael", "ibizan_hound", "irish_setter", "irish_terrier", "irish_water_spaniel",
    "irish_wolfhound", "italian_greyhound", "japanese_spaniel", "keeshond", "kelpie",
    "kerry_blue_terrier", "komondor", "kuvasz", "labrador_retriever", "lakeland_terrier",
    "leonberg", "lhasa", "malamute", "malinois", "maltese_dog", "mexican_hairless",
    "miniature_pinscher", "miniature_poodle", "miniature_schnauzer", "newfoundland",
    "norfolk_terrier", "norwegian_elkhound", "norwich_terrier", "old_english_sheepdog",
    "otterhound", "papillon", "pekinese", "pembroke", "pomeranian", "pug", "redbone",
    "rhodesian_ridgeback", "rottweiler", "saint_bernard", "saluki", "samoyed", "schipperke",
    "scotch_terrier", "scottish_deerhound", "sealyham_terrier", "shetland_sheepdog",
    "shih-tzu", "siberian_husky", "silky_terrier", "soft-coated_wheaten_terrier",
    "staffordshire_bullterrier", "standard_poodle", "standard_schnauzer", "sussex_spaniel",
    "tibetan_mastiff", "tibetan_terrier", "toy_poodle", "toy_terrier", "vizsla", "walker_hound",
    "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier", "whippet",
    "wire-haired_fox_terrier", "yorkshire_terrier"
]

st.sidebar.markdown('<h1 style="color:#f5c793; font-size: 36px;">KNOW YOUR BREED,<br>KNOW YOUR DOG</h1>', 
                    unsafe_allow_html=True)


st.sidebar.info("120 Breeds, Countless Possibilities: Identify Your Dog\'s True Nature!")

# Main UI
st.title(":dog: **The Art of Dog Vision**")
st.markdown(""" 
Welcome to the **Dog Breed Prediction App**! Upload a photo of your furry friend, and let the magic happen! 
- :camera: **Upload** an image. 
- :dog2: **Discover** your dog's breed. 
- :chart_with_upwards_trend: **Explore** predictions visually. 
""")

# Expander for more info
with st.expander("ℹ️ About This App"):
    st.write("""
             This application addresses the challenge of multi-class image classification for 120 different 
             dog breeds, the project extends its reach into the realm of real-world applications, showcasing 
             the versatility of machine learning. 
             It uses a pre-trained deep learning model to predict dog breeds and is built with **TensorFlow**, 
             **Streamlit**, and a touch of creativity!
    """)

uploaded_file = st.file_uploader("Upload a dog image (JPG, PNG, JPEG):")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Preprocess and predict
    input_data = preprocess_image(tmp_file_path)
    input_data = tf.expand_dims(input_data, axis=0)

    # Make the prediction
    predictions = model.predict(input_data)

    # Get top 5 predictions
    top_5_indices = predictions[0].argsort()[-5:][::-1]
    top_5_breeds = [CLASS_NAMES[i] for i in top_5_indices]
    top_5_confidences = predictions[0][top_5_indices]

    # Fetch the description for the breed (defaulting to "No description available." if not found)
    breed_description = dog_descriptions.get(top_5_breeds[0], ["No description available."])

    # Format the breed name: replace underscores with spaces and capitalize the first letter of each word
    formatted_breed_name = " ".join([word.capitalize() for word in top_5_breeds[0].split("_")])

    # Display the formatted breed name in the sidebar header
    st.sidebar.header(f"About {formatted_breed_name}!")

    # Display all lines from the description list
    for line in breed_description:
        st.sidebar.write(f"• {line}")


    # Highlight primary prediction
    st.subheader(f"Predicted Breed: **{formatted_breed_name}** :dog:")
    st.metric(label="Confidence", value=f"{top_5_confidences[0] * 100:.2f}%")

    st.markdown("### :sparkles: Top 5 Predictions")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=top_5_confidences,
        y=top_5_breeds,
        palette="winter",
        ax=ax
    )
    ax.set_title("Top 5 Predicted Breeds", fontsize=16, fontweight="bold")
    ax.set_xlabel("Confidence (%)", fontsize=14)
    ax.set_ylabel("Breed", fontsize=14)

    # Annotate percentages on the bars
    for i, (breed, confidence) in enumerate(zip(top_5_breeds, top_5_confidences)):
        ax.text(
            confidence + 0.01,
            i,
            f"{confidence * 100:.2f}%",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold"
        )

    st.pyplot(fig)

else:
    st.info(":arrow_up: Upload an image to get predictions!")
