import streamlit as st
import torch
import numpy as np
from src.model.AutoEncoder import ConvAutoencoder 
import pickle
import os
import imageio
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial import distance
import random
class page1:
    """Home page of the app"""

    def __init__(self):
        # load in the entire training set (FIND BETTER WAY TO DO THIS)
        self.train_images = self.load_images_from_folder("src/data/train_images")
        self.train_images.extend(self.load_images_from_folder("src/data/test_images")) 
        self.train_images =  self.shuffle_data(self.train_images)
        # load in kmeans model
        with open('src/model/kmeans_model.pkl', 'rb') as file:
            self.kmeans = pickle.load(file)
        # load in autoencoder model
        self.autoencoder = ConvAutoencoder()
        self.autoencoder.load_state_dict(torch.load('src/model/autoencoder_state_dict.pth'))
        self.autoencoder.eval()
        # load in predicted clusters and encoded images
        self.clusters = np.loadtxt("src/data/clusters.txt", dtype=int)
        self.encoded_images_np = np.loadtxt("src/data/encoded_images_np.txt")

    def load_images_from_folder(self,folder):
        images = []
        for filename in os.listdir(folder):
            img = imageio.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
        return images
    def shuffle_data(self,data):
        random.seed(42)
        for i in range(len(data)-1, 0, -1):
            j = random.randint(0, i)
            data[i], data[j] = data[j], data[i]
        return data
    def preprocess_image(self,uploaded_file):
        # convert to PIL Image (slightly diffferes from how we did in the notebook image = Image.fromarray(image_matrix))
        image = Image.open(uploaded_file).convert('RGB')

        # define the transformation to resize and then normalize the image
        transform = transforms.Compose([
            transforms.Resize((96, 96)),        
            transforms.ToTensor(),              
            transforms.Normalize(mean=[0.4470, 0.4397, 0.4056], 
                                std=[0.2605, 0.2566, 0.2705])
        ])

      
        image_tensor = transform(image)
        return image_tensor


    def show(self):

        st.title("What the What?")
        

        #image input widget
        input_image = st.file_uploader("Upload a file", type=("png", "jpg", "jpeg"))

        if input_image is not None:
            autoencoder = self.autoencoder
            kmeans = self.kmeans
            clusters = self.clusters
            encoded_images_np = self.encoded_images_np
            train_images = self.train_images
            st.image(input_image, caption="Uploaded Image", use_column_width=True)
            #preprocess image
            new_image_tensor = self.preprocess_image(input_image)

            # run the image through the auto encoder
            encoded_new_image = autoencoder.encoder(new_image_tensor)
            # flatten and convert to numpy array
            encoded_new_image_flat = encoded_new_image.view(-1).detach().numpy().astype(np.float32).reshape(1,-1)  
            # predict cluster
            prediction = kmeans.predict(encoded_new_image_flat)
            

            # get all data points in each cluster
            input_image_cluster = prediction[0]
            st.write(input_image_cluster)
            # find all indexes of data points in input_image_cluster 
            input_image_cluster_matches = np.where(clusters == input_image_cluster)[0]
            # filter train_images (used to display images)
            train_images_filtered = [train_images[i] for i in input_image_cluster_matches]
            # filter preprocess images (used to find the most similar photos using euclidean distance)
            preprocessed_train_images_filtered = np.array([encoded_images_np[i] for i in input_image_cluster_matches])
            # flatten the new image
            encoded_new_image_flat  = encoded_new_image_flat.ravel()
            # calculate euclidean distances
            distances = [distance.euclidean(encoded_new_image_flat, img_flat) 
                        for img_flat in preprocessed_train_images_filtered]
            # find and retrieve the two closest images
            closest_indices = np.argsort(distances)[:2]
            closest_images = [train_images_filtered[i] for i in closest_indices]

            # Convert arrays to PIL Images if they are numpy arrays
            if isinstance(input_image, np.ndarray):
                image = Image.fromarray(image)
            closest_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in closest_images]

            # Display the original image and the two closest images
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(input_image, caption='Original Image', use_column_width=True)

            with col2:
                st.image(closest_images[0], caption='Closest Image 1', use_column_width=True)

            with col3:
                st.image(closest_images[1], caption='Closest Image 2', use_column_width=True)
                        



