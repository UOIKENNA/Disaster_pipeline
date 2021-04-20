# import libraries
import nltk
import re
nltk.download('punkt')
nltk.download('wordnet')
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import gmean
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
import pickle
import sys

def load_data(database_filepath):
    """load data from database
    Args:
        database_filepath => filepath of database to be used
    Returns:
        X => explanatory variable
        Y => predictive variable
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('Disasters', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    """Tokenizes a df text series
    Args:
        df text series object
    Returns:
        clean token list
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'url_place_holder_string')
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(w).lower().strip()
                    for w in tokens]
    return clean_tokens

def build_model():
    """Build a multi-output classification model
    Returns:
        pipeline => multi-output classifier pipeline
    """
    pipeline1 = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))

        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])


    parameters_grid = {'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
              'classifier__estimator__n_estimators': [10, 20, 40]}

    simple_pipeline_cv = GridSearchCV(pipeline1, param_grid=parameters_grid, scoring='f1_micro', n_jobs=-1)

    return simple_pipeline_cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    inputs
        model
        X_test
        y_test
        category_names
    output:
        scores
    """

    y_pred = model.predict(X_test)

    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(y_test.iloc[:, i].values, y_pred[:, i])))


def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    pickle.dump(model, open('model.pkl', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
