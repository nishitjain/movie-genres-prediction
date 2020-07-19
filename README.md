## Movie Genres Prediction
This project used multi label classification techniques to predict movie genres using movie posters.

Project Structure:

 - **[train.py](https://github.com/nishitjain/movie-genres-prediction/blob/master/train.py)** : This file can be used to train and save models. Please check the required command line arguments before running the file using --help argument. [python train.py --help]
 - **[predict.py](https://github.com/nishitjain/movie-genres-prediction/blob/master/predict.py)**: This file can be used to make predictions on new movie poster images and the trained model. Check required arguments using --help argument. [python predict.py --help]
 - **[util.py](https://github.com/nishitjain/movie-genres-prediction/blob/master/util.py)**: Contains all the required utility functions. 

NOTE: Refer requirements.txt and pip-install it before running the code.

 The dataset zip can be downloaded from **[here](https://drive.google.com/file/d/1bi1VQ0nCZBsa8V27oQKfwlvAQAzM3DpA/view?usp=sharing)**. The zip files contains one folder with all the images and one csv file paths of which needs to be passed to while running train.py as a command line arguments.
