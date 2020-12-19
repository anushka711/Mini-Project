import streamlit as st
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import sklearn

category=['bulldog','maltese','dachshund','germanshepherd','siberianhusky']

model = pickle.load(open('img_model.pkl','rb'))

def func(url):
	flat_data = []
	img = imread(url)
	img_resized = resize(img, (150,150,3))
	flat_data.append(img_resized.flatten())
	flat_data = np.array(flat_data)
	plt.imshow(img_resized)
	y_out = model.predict(flat_data)
	y_out = category[y_out[0]]
	return y_out

def main():
	page_bg_img = '''
	<style>
	body {
	background-image: url("https://images.unsplash.com/photo-1548199973-03cce0bbc87b?ixid=MXwxMjA3fDB8MHxzZWFyY2h8Mnx8ZG9nc3xlbnwwfHwwfA%3D%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=400&q=60");
	background-size: cover;
	}
	</style>
	'''

	st.markdown(page_bg_img, unsafe_allow_html=True)
	st.title("Image Classifier - Breeds of Dogs")
	st.write("Bulldog | maltese | dachshund | germanshepherd | siberianhusky")
	st.write("Enter URL of the image (jpeg):")
	url_input = st.text_input("Label")
	predict_button = st.button("Predict")
	if predict_button:
		result = func(url_input)
		st.write("Breed: {}".format(result))

if __name__=="__main__":
	main()
