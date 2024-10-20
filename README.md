# Artist Recommender
A streamlit app that recommends artists based on artists submitted by the user
Try the app [here](https://dansartistrecommender.streamlit.app/)

## Usage
Search for artists by entering an artist's name in the field. Upon pressing enter or clicking off the box, a message will be displayed with the name of the matching artist, if found. If the artist is not found, an error message will show - please try a different search term.

When all 3 artists are found, click the 'Get Recommendations' button. The app will return the names and images of 3 similar artists, along with a score representing how close of a match they are.

## Method
### Overview
Matches are based on the similarity of the user submitted tags. I chose this as it is a collaborative filtering based approach as opposed to a content filtering. In my opinion the data that can be collected on the content of a song (lyrics, key, BPM) are much less useful as a measure of similarity. In addition, the metrics Spotify provide (energy, acousticness, danceability etc.) are often innacurate - not to mention that using Spotify data for machine learning is against their terms of service.

### Artist dataset
The dataset that matches are pulled from was collected from the top 1000 artists (as of October 18th 2024) through the last.fm api. From there the top 5 user submitted tags for each are collected (also through the last.fm api) to serve as the basis of the matching algorithm.

The tags are encoded into the database through sklearn's TF-IDF vectoriser. I chose a TF-IDF approach to emphasise the nuance and uniqueness of each artist, giving more unique tags a larger weight. This should help provide matches that better capture the details of each artist - not just their genre. However, this has some drawbacks as for the most popular genres, the genre tag is almost entirely ignored and sometimes provides matches of different genre. This is not necessarily an issue - a user may be open to exploring artists of different genres if they have the same quirks as the artists they submitted.

The result of the dataset is a table where each artist is given a score for each tag (mostly zero) that represents the characteristics of the artist according to user submitted tags.

### Searching for artists
Artists are searched for by running a search through the last.fm api and returning the first match, if any. The success message displays the name of the found artist for the user to check if it's what they were looking for. The error message displays if there are no results (`len(results) == 0`)

From there the top 5 tags of each artist are collected and encoded using the same vectoriser as before (to prevent mismatched tags). Again another drawback arises as not all of the tags will be present in the original encoding. In some cases none of the top 5 tags will be present in the original dataset - so this artist is essentially ignored in the matchmaking process. In future versions I may update the algorithm to skip tags that are not present in the original dataset. 

The 3 chosen artists are saved in a dataframe identical to the artist database - each artist given a score in each tag to represent the characteristics of said artist.

### Calculating matches
Each row in the artist database table can be assumed as a vector. The entries for the 3 chosen artists are removed from the dataset if present to prevent being recommended a submitted artist. The 'vectors' for these 3 artists are averaged to give a resulting 'preference vector' which represents the users preference in artists.

This preference vector is used to transform the artist dataset to give the cosine similarity between each artist's vector and the users 'preference vector'. The transformed table is sorted and the top 3 artists are returned as matches. The image for each artist is pulled using the Spotify api and the score is simply `1 - cosine distance` as a percentage.

# Technologies used
- pandas - Data manipulation
- scikit-learn - TF-IDF vectoriser
- pylast - Last.fm API access
- spotipy - Spotify API access


