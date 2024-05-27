import streamlit as st
import os
import pickle
import numpy as np
import tensorflow as tf
import shutil
import cv2
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image
import subprocess

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

os.makedirs("displayed_images/images", exist_ok=True)
input_folder = 'displayed_images\images\images'
output_folder = 'displayed_images\images\png_images'
desired_size = (1080, 1440)

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# clear the displayed_images/images/ folder
folder_path = 'displayed_images/images/images'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# clear the displayed_images/images/png_images folder
folder_path = 'displayed_images/images/png_images'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)

        # create the necessary directories for the copied image files
        os.makedirs(os.path.dirname('displayed_images/images/' + filenames[indices[0][0]]), exist_ok=True)

        with col1:
            st.image(filenames[indices[0][0]])
            shutil.copy(filenames[indices[0][0]], os.path.join('displayed_images/images', filenames[indices[0][0]].split('/')[-1]))
        with col2:
            # create the necessary directories for the copied image files
            os.makedirs(os.path.dirname('displayed_images/images/' + filenames[indices[0][1]]), exist_ok=True)

            st.image(filenames[indices[0][1]])
            shutil.copy(filenames[indices[0][1]], os.path.join('displayed_images/images', filenames[indices[0][1]].split('/')[-1]))
        with col3:
            # create the necessary directories for the copied image files
            os.makedirs(os.path.dirname('displayed_images/images/' + filenames[indices[0][2]]), exist_ok=True)

            st.image(filenames[indices[0][2]])
            shutil.copy(filenames[indices[0][2]], os.path.join('displayed_images/images', filenames[indices[0][2]].split('/')[-1]))
        with col4:
            # create the necessary directories for the copied image files
            os.makedirs(os.path.dirname('displayed_images/images/' + filenames[indices[0][3]]), exist_ok=True)

            st.image(filenames[indices[0][3]])
            shutil.copy(filenames[indices[0][3]], os.path.join('displayed_images/images', filenames[indices[0][3]].split('/')[-1]))
        with col5:
            # create the necessary directories for the copied image files
            os.makedirs(os.path.dirname('displayed_images/images/' + filenames[indices[0][4]]), exist_ok=True)

            st.image(filenames[indices[0][4]])
            shutil.copy(filenames[indices[0][4]], os.path.join('displayed_images/images', filenames[indices[0][4]].split('/')[-1]))

        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg"):  # Check for JPEG images
                # Read the image
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                # Convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Threshold the grayscale image to create a mask where the background is white
                _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

                # Invert the mask to get the background as black
                mask_inv = cv2.bitwise_not(mask)

                # Convert the original image to BGRA (Blue, Green, Red, Alpha)
                image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

                # Split the channels
                b, g, r, a = cv2.split(image_bgra)

                # Use the inverted mask to set the alpha channel of the image
                a = cv2.bitwise_and(mask_inv, a)

                # Merge the channels back
                image_bgra = cv2.merge((b, g, r, a))

                # Resize the image to the desired size
                resized_image = cv2.resize(image_bgra, desired_size, interpolation=cv2.INTER_AREA)

                # Save the result
                output_path = os.path.join(output_folder, filename[:-4] + '.png')
                cv2.imwrite(output_path, resized_image)

        if st.button(' VTR FOR MALE'):
            subprocess.run(['python', 'male.py'])

        if st.button(' VTR FOR FEMALE'):
            subprocess.run(['python', 'female.py'])