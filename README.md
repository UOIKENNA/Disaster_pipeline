# Disaster_pipeline Project
The mission of this project is to apply data science skills to a dataset provided by figure8 which contains thousands of real time disaster response messages sent to various orgnizations during disasters. We want to build a model that classifies disaster messages for api usage.

#Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Open another terminal, run env|grep WORK. You'll see the following output WORKSPACEDOMAIN=udacity-student-workspaces.com WORKSPACEID=view6914b2f4 Now, use the above information to open https://view6914b2f4-3001.udacity-student-workspaces.com/ (general format - https://WORKSPACEID-3001.WORKSPACEDOMAIN/)


# Table of Contents

1. [Installation](https://github.com/UOIKENNA/Disaster_pipeline#Installation)
2. [Project Motivation](https://github.com/UOIKENNA/Disaster_pipeline#Project)
3. [File Descriptions](https://github.com/UOIKENNA/Disaster_pipeline#File)
4. [Results](https://github.com/UOIKENNA/Disaster_pipeline#Results)
5. [Licensing, Authors and Acknowledgements](https://github.com/UOIKENNA/Disaster_pipeline#licensing)


# Installation
The code should run without issues using Python 3.*. All the necessary libraries were imported using the Anaconda distribution.

# Project Motivation
During disasters, lots of messages are sent across; a lot of these messages get lost thanks to the sheer volume of the messages. The idea behind this project is to ensure that as many messages as possible get to the required destination and at the time sent

# File Descriptions
The files are split into 3 different parts;
1) Data Processing: We build an ETL (Extract, Transform, and Load) pipeline that processes messages and category data from CSV file, and load them into an SQLite database - which our ML pipeline will then read from to create and save a multi-output supervised learning model.

2) Machine Learning Pipeline: We split the data into a training set and a test set. Then, create an ML pipeline that uses NLTK, as well as GridSearchCV to output a final model that predicts message classifications for the 36 categories (multi-output classification)

3) Web development: We deploy a web application that classifies messages.

# Results
![image](https://user-images.githubusercontent.com/40573980/116608630-58985c00-a92b-11eb-901f-a1de0c7a2afe.png)
![image](https://user-images.githubusercontent.com/40573980/116609966-e759a880-a92c-11eb-92c7-75720cb5148d.png)
![image](https://user-images.githubusercontent.com/40573980/116610075-05bfa400-a92d-11eb-9a05-c7ff5c1c0432.png)



# Licensing, Authors and Acknowledgements
Credit to Udacity and Figure8 for the dataset provided. Also I want to give credit to https://github.com/deogakofi and https://github.com/OliviaCrrbb for their assistance. This project is open to contributions
