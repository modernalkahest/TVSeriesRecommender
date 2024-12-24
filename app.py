import pandas as pd
import numpy as np
from warnings import filterwarnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,request,jsonify

app = Flask(__name__)

filterwarnings('ignore')
#loading the data set
anime = pd.read_csv('Anime.csv')
Ratings = anime[['Name','Type','Studio','Rating','Tags']]

#Finding null values and replacing them
Ratings['Studio']=Ratings['Studio'].fillna('Unknown')
Ratings['Tags']=Ratings['Tags'].fillna('Unknown')

#Encoding the categorical variable Studio
encoder = dict([(j,i) for i,j in enumerate(Ratings['Studio'].value_counts().index)])
Ratings.set_index('Name',inplace=True)
Ratings['Studio'] = Ratings.apply(lambda row: encoder[row['Studio']],axis=1)

#Encoding the categorical variable Type
Type_encoder = dict([(j,i) for i,j in enumerate(Ratings['Type'].unique())])
Ratings['Type'] = Ratings.apply(lambda row: Type_encoder[row['Type']],axis=1)


#Taking user input
anime_watched = input("What was the name of the latest anime you watched? ")

#finding closest matches to user input
def cosine_sim(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

@app.route('/recommend', methods=['GET'])
def recommend():
    anime_watched = request.args.get('anime_watched')
    if not anime_watched:
        return jsonify({'error': 'Please provide the name of the anime you watched'}), 400

    if anime_watched not in Ratings.index:
        matches = anime['Name'].apply(lambda x: cosine_sim(anime_watched, x))
        matches.index = anime['Name']
        matches = matches.sort_values(ascending=False)
        matches = matches.to_frame()
        match_list = list(enumerate(matches.head().index))
        match_list_dict = dict(match_list)
        anime_watched = match_list[0][1]  # Automatically select the first match

    # Isolating the features on which recommendations should be given
    Anime_SR = Ratings[['Studio', 'Rating']]

    # Finding similarities between user input title and existing database
    Cos_Similarity = Anime_SR.apply(lambda row: np.dot(Anime_SR.loc[anime_watched], row) / (np.linalg.norm(Anime_SR.loc[anime_watched]) * np.linalg.norm(row)), axis=1)

    # Converting to Dataframe
    Cos = Cos_Similarity.to_frame()
    Anime_Tags = Ratings['Tags']

    # Adding columns to the dataframe
    Cos.columns = ['Cosine Similarity']
    Cos['Tag Similarity'] = Anime_Tags.apply(lambda row: cosine_sim(Anime_Tags.loc[anime_watched], row))

    # Sorting recommendations by Cosine similarity
    Recommendation = Cos[(Cos['Cosine Similarity'] > 0.5) & (Cos['Tag Similarity'] > 0.5)].sort_values(by='Tag Similarity', ascending=False)

    # Getting top recommendations from the entire dataframe
    recommendation_list = list(Recommendation.index)
    if anime_watched in recommendation_list:
        recommendation_list.remove(anime_watched)

    recommendations = [i for i in recommendation_list if cosine_sim(anime_watched, i) == 0]

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)