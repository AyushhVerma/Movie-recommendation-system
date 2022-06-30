import numpy as np
import pandas as pd

### Data preprocessing

df = pd.read_csv(r'C:\Users\ayush\Desktop\MP\u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
# print(df.head())

movies = pd.read_csv(r'C:\Users\ayush\Desktop\MP\Movie_Id_Titles')
# print(movies.head())

DF = pd.merge(df, movies, on='item_id')
# print(DF.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings = pd.DataFrame(DF.groupby('title')['rating'].mean())
# print(ratings.head())

ratings['num_of_votes'] = pd.DataFrame(DF.groupby('title')['rating'].count())
# ratings['num_of_votes'].hist(bins=70)

sns.jointplot(x='rating', y='num_of_votes', data=ratings, alpha=0.5)

### content-based recommendation system

moviegr = DF.pivot_table(index='user_id', columns='title', values='rating')
ratings.sort_values('num_of_votes', ascending=False).head(10)

def get_recommendation(movie):
    user_ratings = moviegr[movie]
    similarity = moviegr.corrwith(user_ratings)
    corr_ = pd.DataFrame(similarity, columns=['Correlation'])
    corr_.dropna(inplace=True)
    # print(similarity)
    corr_ = corr_.join(ratings['num_of_votes'])
    return corr_[corr_['num_of_votes']>100].sort_values('Correlation', ascending=False).head(10)


print(get_recommendation('Fargo (1996)'))
print(get_recommendation('Scream (1996)'))