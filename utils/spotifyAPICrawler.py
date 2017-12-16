import pandas as pd
import spotipy


def get_auth_token(username, client_id, client_secret, redirect_uri):
    token = spotipy.util.prompt_for_user_token(username,
                                               client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri)

    return token

def querry_spotify(filepath, token, verbose=False, ):
    """
    Function querries the spotifyAPI, by using the artist name and the song title
    :param filepath: str | filepath
    :return: pd.df | dataframe with the audio features from the spotifyAPI
    """
    # search for track + name
    songs = pd.read_csv(filepath)
    AudioFeatures2 = pd.DataFrame()
    i = 0
    media_ids_na = []
    for index, row in songs.iterrows():
        name = row['title']
        title_short = row['title_short']
        artist = row['artist']
        bpm = row['bpm']
        media_id = row['id']
        try:
            spotify = spotipy.Spotify(auth=token)
            results = spotify.search(
                q='track:' + title_short + ' artist:' + artist, type='track',
                limit=1)
            uri = results['tracks']['items'][0]['uri']
            spotify_name = results['tracks']['items'][0]['name']
            album_name = results['tracks']['items'][0]['album']['name']
            sp_artist = results['tracks']['items'][0]['artists'][0]['name']
            audio_feat = spotify.audio_features(str(uri))
            x = audio_feat[0]
            x.pop('analysis_url')
            x.pop('track_href')
            x['spotify_name'] = spotify_name
            x['spotify_album_name'] = album_name
            x['deezer_name'] = name
            x['deezer_artist'] = artist
            x['deezer_bpm'] = bpm
            x['spotify_artist'] = sp_artist
            x['media_id'] = media_id
            AudioFeatures2 = AudioFeatures2.append(x, ignore_index=True)
        except:
            media_ids_na.append(media_id)
            i += 1
            if verbose:
                print(i)
            continue
    return AudioFeatures2