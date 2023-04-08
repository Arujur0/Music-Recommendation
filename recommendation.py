import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics

ID = "3dce0b0ef6a442c9878c48e689d70df7"
SECRET = "9a97c11430d04273ad190cab400479eb"
client_credentials_manager = SpotifyClientCredentials(client_id=ID, client_secret=SECRET)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

def analyze_playlist(creator, playlist_id): 
    # Create empty dataframe
    playlist_features_list = ["artist", "album", "track_name",  "track_id","danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms", "time_signature"]
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    # Loop through every track in the playlist, extract features and append the features to the playlist df
    
    playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    for track in playlist:
        # Create empty dict
        playlist_features = {}
        # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]
        
        # Get audio features
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[4:]:
            playlist_features[feature] = audio_features[feature]
        
        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
        
    return playlist_df

def get_dataframe(df, links, user_ids):
    for key in links:
        fd = analyze_playlist(user_ids[key], links[key])
        fd['Emotion'] = key
        df = df.append(fd)
    return df


def recommend_song_by_mood(df, songs_df, mood_label, numerical_features):
    mood_label = int(mood_label)
    if mood_label == 1:
        mood_songs = songs_df[songs_df['Emotion'] == 0]
    elif mood_label == 5:
        mood_songs = songs_df[songs_df['Emotion'] == 2]
    else:
        mood_songs = songs_df[songs_df['Emotion'] == mood_label]
    distances, indices = nei.kneighbors(mood_songs[numerical_features].mean(axis=0).values.reshape(1, -1))
    mood_song_scores = pd.DataFrame({
        'song_id': df.index[indices[0]],
        'score': distances[0],
    }).sort_values('score', ascending=True)
    
    song_id = mood_song_scores.head(1)['song_id'].values[0]
    x = df.loc[song_id]
    return x, song_id

acc = []
df = pd.DataFrame()
features = ['danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence']
df = pd.read_csv('readthisinsteadofscrapping.csv')
df = df.drop(columns=['Unnamed: 0'])
dfa = df.drop(columns=['artist', 'album', 'track_name', 'track_id', 'key', 'mode', 'duration_ms', 'time_signature'])

scale = StandardScaler()
dfa[features] = scale.fit_transform(dfa[features])
X = dfa[features]
y = dfa['Emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


for i in range(1,60):
    neigh = KNeighborsClassifier(n_neighbors = i, metric='euclidean', weights='uniform').fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat)) 

plt.figure(figsize=(10,6))
plt.plot(range(1,60),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

nei = KNeighborsClassifier(n_neighbors=acc.index(max(acc)), metric="euclidean", weights="uniform")
nei.fit(X_train, y_train)
