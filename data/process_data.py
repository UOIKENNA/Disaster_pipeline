# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_file_path, categories_file_path):
    """Load data from csv files and merge both datasets on ID
    Args:
        messages_file_path => csv file for messages
        categories_file_path => csv file for categories
    Returns:
        df => Dataframe messages and categories merged on ID
    """
    messages = pd.read_csv(messages_file_path)
    categories = pd.read_csv(categories_file_path)
    df = messages.merge(categories, on = 'id')
    return df

# clean categories and merge to messages
def clean_data(df):
    """Clean categories and merge to messages
    Args:
        df => DataFrame of merged categories and messages csv files
    Returns:
        df => Dataframe of cleaned categories and dropped duplicates
    """
    categories = pd.Series(df.categories).str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:]).values

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    df = pd.concat([df,pd.get_dummies(df['genre'])],axis=1)
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    df = df.drop(['genre','social'],axis=1)

    return df


def save_data(df, database_filename):
    """Save dataframe to sqlite engine
    Args:
        df => DataFrame of merged categories and messages csv files
        database_filename => filename for db engine as string
    Returns:
        None
    """
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df.to_sql('disaster', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
