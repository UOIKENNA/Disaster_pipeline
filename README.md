# Disaster_pipeline Project
The mission of this project is to apply data science skills to a dataset provided by figure8 which contains thousands of real time disaster response messages sent to various orgnizations during disasters. We want to build a model that classifies disaster messages for api usage.

# Table of Contents

1. [Installation](https://github.com/UOIKENNA/Disaster_pipeline#Installation)
2. [Project Motivation](https://github.com/UOIKENNA/Disaster_pipeline#Project)
3. [File Descriptions](https://github.com/UOIKENNA/Disaster_pipeline#File)
4. [Licensing, Authors and Acknowledgements](https://github.com/UOIKENNA/Disaster_pipeline#licensing)

# Installation
The code should run without issues using Python 3.*. All the necessary libraries were imported using the Anaconda distribution.

# Project Motivation
During disasters, lots of messages are sent across; a lot of these messages get lost thanks to the sheer volume of the messages. The idea behind this project is to ensure that as many messages as possible get to the required destination and at the time sent

# File Descriptions
The files are split into 3 different parts;
1) Data Processing: We build an ETL (Extract, Transform, and Load) pipeline that processes messages and category data from CSV file, and load them into an SQLite database - which our ML pipeline will then read from to create and save a multi-output supervised learning model.

2) Machine Learning Pipeline: We split the data into a training set and a test set. Then, create an ML pipeline that uses NLTK, as well as GridSearchCV to output a final model that predicts message classifications for the 36 categories (multi-output classification)

3) Web development: We deploy a web application that classifies messages.

# Licensing, Authors and Acknowledgements
Credit to Udacity and Figure8 for the dataset provided. Also I want to give credit to https://github.com/deogakofi and https://github.com/OliviaCrrbb for their assistance. This project is open to contributions
