# import dependencies
import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from config import db_password
import psycopg2
import time

# initate file directroy
file_dir ="C:/Users/michael1005/Desktop/UCB_Data_Analytics/Module 8/Module-8"

# below is for function start ------------------------------------------------------------------
def etl_process(wiki_json = "wikipedia.movies.json",kaggle_meta_data = "movies_metadata.csv",ratings_data = "ratings.csv"):
    #Open file
    with open(f'{file_dir}/{wiki_json}', mode='r') as file:
        wiki_movies_raw = json.load(file)

    #put movie data in DFs
    kaggle_metadata = pd.read_csv(f'{file_dir}/{kaggle_meta_data}')
    ratings = pd.read_csv(f'{file_dir}/{ratings_data}')

    # make wiki data into DF
    wiki_movies_df = pd.DataFrame(wiki_movies_raw)
    
    # list of columns (193)
    wiki_movies_df.columns.tolist()

    #filter expression by directors and IMBD ratings
    wiki_movies = [movie for movie in wiki_movies_raw
                if ('Director' in movie or 'Directed by' in movie)
                    and 'imdb_link' in movie
                    and 'No. of episodes' not in movie]

    #put wiki_movies into DF
    wiki_movies_df = pd.DataFrame(wiki_movies)

    #function to clean the movie
    def clean_movie(movie):
                
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {}
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune–Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
        if len(alt_titles) > 0:
            movie['alt_titles'] = alt_titles

        #merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
                
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')
        
        return movie

    # clean movie list and df
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    wiki_movies_df = pd.DataFrame(clean_movies)

    #change imdb_id
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
    # drop duplicate imdb column
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)

    # get rid of mostly NaN columns
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    # drop box office rows with no data
    box_office = wiki_movies_df['Box office'].dropna()

    #box office lambda function
    box_office[box_office.map(lambda x: type(x) != str)]
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

    # reg expression for normalizing box office data
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    box_office.str.contains(form_one, flags=re.IGNORECASE).sum()

    # reg expression for other data type of box office
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
    box_office.str.contains(form_two, flags=re.IGNORECASE).sum()

    #create exceptions if matches either
    matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)
    box_office[~matches_form_one & ~matches_form_two]

    # repalce box office
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    #function for parse dollar values
    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan

        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " million"
            s = re.sub(r'\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a million
            value = float(s) * 10**6

            # return value
            return value

        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " billion"
            s = re.sub(r'\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a billion
            value = float(s) * 10**9

            # return value
            return value

        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

            # remove dollar sign and commas
            s = re.sub(r'\$|,','', s)

            # convert to float
            value = float(s)

            # return value
            return value

        # otherwise, return NaN
        else:
            return np.nan

    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    # form budget parse, map to strings and mevoe $ and hyphens
    budget = wiki_movies_df['Budget'].dropna()
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    #budget matches 
    matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
    budget[~matches_form_one & ~matches_form_two]

    # string replace budget
    budget = budget.str.replace(r'\[\d+\]\s*', '')
    budget[~matches_form_one & ~matches_form_two]

    #replace wiki movies df
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    # drop budget and box office original columns
    wiki_movies_df.drop('Budget', axis=1, inplace=True)
    wiki_movies_df.drop('Box office', axis=1, inplace=True)

    #parse release date with NA
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    # four forms for dates
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'

    #extract strings release date
    release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)

    #put parsed release date into new column
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    #drop old column
    wiki_movies_df.drop('Release date', axis=1, inplace=True)

    #variable for NA for urnning time
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    # check number captured by reg exp
    running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()

    # check format for remaining to parsse
    running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]

    #extract by groups
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    #fill Nas with 0s
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

    # put parsed running time into new column
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

    #drop old running time
    wiki_movies_df.drop('Running time', axis=1, inplace=True)

    # check for bad data (true/false)
    kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]

    #remove adult movies and drop the column
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

    # keep only true 
    kaggle_metadata['video'] == 'True'

    #assign back to the metadata
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

    # convert numerical data
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')

    #convert release date to datetime
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

    # change ratings to date time data
    pd.to_datetime(ratings['timestamp'], unit='s')

    # put ratings datetime into ratings DF
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    # merging two DFs
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

    #check for missing title - can drop wikipedia titles, since no missing
    movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]

    #get rid of bad data - start by indexing row
    movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index

    # remove the row (by removing index)
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)

    # convert language lists to tuples to read
    movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)

    # drop columns we dont want
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

    #function to fill missing data
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)

    # fill missing data with wiki data
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    #drop vidoe column
    movies_df.drop(columns=['video'], inplace=True)

    # reorder columns
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                        'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                        'genres','original_language','overview','spoken_languages','Country',
                        'production_companies','production_countries','Distributor',
                        'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                        ]]

    # rename columns
    movies_df.rename({'id':'kaggle_id',
                    'title_kaggle':'title',
                    'url':'wikipedia_url',
                    'budget_kaggle':'budget',
                    'release_date_kaggle':'release_date',
                    'Country':'country',
                    'Distributor':'distributor',
                    'Producer(s)':'producers',
                    'Director':'director',
                    'Starring':'starring',
                    'Cinematography':'cinematography',
                    'Editor(s)':'editors',
                    'Writer(s)':'writers',
                    'Composer(s)':'composers',
                    'Based on':'based_on'
                    }, axis='columns', inplace=True)

    # count ratings and rename count as Id and pivot Id to index
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                    .rename({'userId':'count'}, axis=1) \
                    .pivot(index='movieId',columns='rating', values='count')

    # change name to rating _
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]
    # merge ratings with movies df
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    # find missing movises without ratings and fill with 0
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

    # create connection to database
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"

    #databse engine
    engine = create_engine(db_string)

    # potential place to add code to delete old data in movies and ratings (KEEP TABLE)

    #save movies to SQL 
    movies_df.to_sql(name='movies', con=engine, if_exists='replace')

    rows_imported = 0
    # get the start_time from time.time()
    start_time = time.time()
    for data in pd.read_csv(f'{file_dir}/{ratings_data}', chunksize=1000000):
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        if rows_imported == 0:
            data.to_sql(name='ratings', con=engine, if_exists='replace')
        else:
            data.to_sql(name='ratings', con=engine, if_exists='append')
        rows_imported += len(data)

        # add elapsed time to final print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')

# run ETL process function - inputs = 3 files we were given
etl_process('wikipedia.movies.json','movies_metadata.csv','ratings.csv')