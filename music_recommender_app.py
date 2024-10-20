import streamlit as st
import pandas as pd
import sklearn
import pylast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Manage APIs for Last.fm, Spotify
SP_API_KEY = st.secrets['SP_API_KEY']
SP_API_SECRET = st.secrets['SP_API_SECRET']
LASTFM_API_KEY = st.secrets['LFM_API_KEY']
LASTFM_API_SECRET = st.secrets['LFM_API_SECRET']

network = pylast.LastFMNetwork(api_key=LASTFM_API_KEY,api_secret=LASTFM_API_SECRET)
client_credentials_manager = SpotifyClientCredentials(client_id=SP_API_KEY,client_secret=SP_API_SECRET)    
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# Initialise lists for searched and confirmed artist, load artist database
artistsFound = [None,None,None]
artistSearch = [None,None,None]
onekartists = pd.read_csv('oneKArtists.csv',index_col=0)


# Initialise TD-IDF vectorizer
vectoriser = TfidfVectorizer()


def prepareRecData():
    # Drop empty rows
    onekartists.dropna(axis=0,inplace=True)
    
    # Remove duplicates
    no_dupes = []
    for item in onekartists:
        if item not in no_dupes:
            no_dupes.append(item)



    # TF-IDF encode artist database and return it as a dataframe    
    tdfidf_matrix = vectoriser.fit_transform(onekartists['Tags'])
    tfidf_df = pd.DataFrame(tdfidf_matrix.toarray(),columns=vectoriser.get_feature_names_out())
    
    return pd.concat((onekartists,tfidf_df),axis = 1).drop(['Tags'],axis=1)
    
def preparePrefVector():
    # Get top 5 tags for each artist
    tags = [artist.get_top_tags(limit = 5) for artist in artistsFound]
    
    # Format tags from pyLast tag object into plain text
    tags_unpk = [[topitem.item.name for topitem in sublist] for sublist in tags]
    tags_format = [[tag.replace(' ','') for tag in sublist] for sublist in tags_unpk]
    tags_cnct = [' '.join(taglist) for taglist in tags_format]
    
    # TF-IDF encode tags and save as dataframe
    tags_tfidf_matrix = vectoriser.transform(tags_cnct)
    tags_tfidf_df = pd.DataFrame(tags_tfidf_matrix.toarray(),columns=vectoriser.get_feature_names_out())
    
    # Find mean of each column to return 'Preference vector'
    return tags_tfidf_df.mean()



def searchArtist(index):
    # Nullify item in list to keep consistency between whats in the text box and whats in the list
    artistsFound[index] = None
    
    # Get search results
    current_search = network.search_for_artist(artistSearch[index])
    current_artist = current_search.get_next_page()
    
    # Validate that any artist has been found
    if len(current_artist) == 0:
        st.error(f'**{artistSearch[index]}** could not be found. Maybe you misspelt their name.')
    else:
        # Store pyLance artist object in list
        found_artist = current_artist[0]
        
        if found_artist in artistsFound:
            st.error("You've already picked this artist") 
        else:
            st.success(f'Found: {found_artist.name}')
            artistsFound[index] = found_artist

def get_artist_image(artist_name):
    # Search for artist image using Spotify api
    results = sp.search(q='artist:' + artist_name, type='artist')
    
    # If artist found, look for items
    if results['artists']['items']:
        
        artist = results['artists']['items'][0]
        
        # If image found, return it
        return artist['images'][0]['url'] if artist['images'] else None
    return None

# Heading and description
st.write('# Artist Recommender')
st.write('Enter 3 artists to get recommendations powered by machine learning')


# Buttons
artistSearch[0] = st.text_input('Artist 1')
if artistSearch[0]:
    searchArtist(0)
artistSearch[1] = st.text_input('Artist 2')
if artistSearch[1]:
    searchArtist(1)
artistSearch[2] = st.text_input('Artist 3')
if artistSearch[2]:
    searchArtist(2)




# Get recommendations buttons
recs_button = st.button('Get recommendations!',disabled=False if all(artistsFound) else True)
if recs_button:
    
    # Load encoded database and preference vector
    artist_db = prepareRecData()
    pref_vector = preparePrefVector()
    # st.write(pref_vector)
    
    # Clean database: drop artists that have been submitted by the user, drop missing values
    searched_names = [artist.name for artist in artistsFound]
    # st.write(f'Dropping artists {searched_names}')
    artists_filtered = artist_db[~artist_db['Name'].isin(searched_names)]
    artists_filtered.dropna(inplace=True)
    
    # Rank matches according to cosine similarity and get top 3
    matches = cosine_similarity([pref_vector],artists_filtered.drop(['Name'],axis=1))[0]
    top3idx = matches.argsort()[-3:][::-1]

    # Get top 3 names and match scores
    top3matches = artists_filtered['Name'].iloc[top3idx]
    top3scores = [round(100 * score,2) for score in matches[top3idx]]
    
    
    
    # Display results
    st.balloons()
    for i,(name,score) in enumerate(zip(top3matches,top3scores)):
        st.subheader(f'{i+1}. {name}',divider='gray')
        # st.write(artists_filtered[artists_filtered['Name'] == name])
        imgcol, scorecol = st.columns([1,1],vertical_alignment='center')
        with imgcol:
            st.image(get_artist_image(name))
        with scorecol:
            st.write(f'## {score}% match')
        