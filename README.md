# Rhizotron Preprocessing
This script processes images from a specified directory to create a high contrast of the Rhizotron using the Red and blue images. The script also perfroms some simple transoformations to the images to correct for defects in the Scanner.

To use this code you need to first download your images off of the Rhiztron computer and save them locally in their original folder structure. The script will then process the images and save the results in the same directory as the original images.
The only option the user needs to define is the path to the data directory. 

############ADD HOW TO SETUP VIRTUAL ENVIRONMENT##################### 
Beefore beginning to run this code make sure you have a a Virtual enviroment setup, to do this open cmd and navigate to the folder you set this Notebook. Next run the following lines as code.

python -m venv venv venv\Scripts\activate ipython kernel install --user --name=venv

The kernal is now installed next install required package numpy pip install numpy
