import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import io

# Load environment variables for Azure AI credentials
load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')

# Authenticate Azure AI Vision client
cv_client = ImageAnalysisClient(endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key))

# Streamlit page setup
st.title("Azure Vision API Image Analysis")

# Upload image using Streamlit
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Define a function to analyze the image
def AnalyzeImage(image_data):
    st.write("Analyzing image...")
    recognized_objects = []

    try:
        # Analyze image using Azure Vision API
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE],
        )

        # Display analysis results
        if result.caption:
            st.write(f"**Caption**: '{result.caption.text}' (Confidence: {result.caption.confidence * 100:.2f}%)")

        if result.dense_captions:
            st.write("**Dense Captions**:")
            for caption in result.dense_captions.list:
                st.write(f"Caption: '{caption.text}' (Confidence: {caption.confidence * 100:.2f}%)")

        if result.tags:
            st.write("**Tags**:")
            for tag in result.tags.list:
                st.write(f"Tag: '{tag.name}' (Confidence: {tag.confidence * 100:.2f}%)")

        image = Image.open(io.BytesIO(image_data))
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        if result.objects:
            st.write("**Objects in the image**:")
            for detected_object in result.objects.list:
                obj_info = f"Object: {detected_object.tags[0].name} (Confidence: {detected_object.tags[0].confidence * 100:.2f}%)"
                st.write(obj_info)
                recognized_objects.append(obj_info)
                r = detected_object.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)

        if result.people:
            st.write("**People detected in the image**:")
            for detected_people in result.people.list:
                r = detected_people.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)

        # Display the modified image with bounding boxes
        st.image(image, caption="Objects detected", use_column_width=True)

        # Save the modified image to a BytesIO object for downloading
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Download button for the modified image
        st.download_button("Download Image with Detected Objects", img_byte_arr, file_name="detected_objects_image.png", mime="image/png")

    except HttpResponseError as e:
        st.error(f"Error: {e.reason} - {e.message}")

    return recognized_objects


# Run the analysis when the image is uploaded
if uploaded_image is not None:
    image_data = uploaded_image.read()
    recognized_objects = AnalyzeImage(image_data)

    # Download functionality for recognized objects
    if recognized_objects:
        download_text = "\n".join(recognized_objects)
        st.download_button("Download Recognized Objects", download_text, file_name="recognized_objects.txt")
