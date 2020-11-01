PROBLEM STATEMENT:
The goal of this project, is to train a deep learning model that will able to recognize a tourist site located in India.The model should be able to recognize arbitrary photos with a high degree of accuracy. It should also be able to warn if it is not confident enough in it's prediction, for example when it is presented with an image class that it was not trained on.

DATASET:
The project is based on a list of 25 popular landmarks, from across India. Since no relevant dataset was found, hence all the images for each landmark was downloaded manually and then filtered out irrelevant/corrupt images. This project uses images from the publicly available platforms and all images have permissive license.
The images is of varying sizes and quality. Also, they are not distributed uniformly among the 25 classes.
* The total number of images in the dataset is 3576
* Training - 2629 	~ 73.5%
* Test - 947 	~  26.5%
* Additionally, there are 14 outlier images, just for fun!

SOFTWARE/LIBRARIES:
* Python 3.7
* PyTorch 1.6
* TorchVision 0.7
* Gradio 1.2
* Image Downloader for Chrome


REFERENCES:
* https://www.greavesindia.co.uk/50-incredible-landmarks-india/
* https://en.wikipedia.org/wiki/List_of_World_Heritage_Sites_in_India
* https://github.com/AKASH2907/indian_landmark_recognition
* https://pytorch.org/docs/stable/torchvision/models.html 


SETUP:
* In order to use the model, a virtual environment using Python 3.7 has to be created.
* Activate the virtual environment
* Run "pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html"
* Run "deploy/deploy.py"
* Open any browser and go to  http://localhost:7860 
* Drag and drop any image on the left side and click “Submit”


# Udacity-Capstone-Landmark_Detection
