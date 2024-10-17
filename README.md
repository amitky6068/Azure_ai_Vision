# Azure Vision API Image Analysis

This project is a web application built using Streamlit that utilizes the Azure Vision API to analyze images. Users can upload images, and the application will return various details such as captions, tags, and detected objects, along with the option to download the modified image with bounding boxes around recognized objects.

## Features

- Upload images in JPG, PNG, or JPEG formats.
- Analyze images using Azure Vision API for:
  - Captions
  - Dense captions
  - Tags
  - Detected objects
  - People
- Display bounding boxes around detected objects and people in the image.
- Download the modified image with bounding boxes.
- Download a text file containing recognized objects.

## Requirements

- Python 3.7 or higher
- Streamlit
- Azure AI Vision SDK
- Python-dotenv
- Pillow

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/azure-vision-image-analysis.git
   cd azure-vision-image-analysis
