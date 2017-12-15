"""
This module converts a list of song_id's from the deezer dataset to names of
the songs that can be used for the test
"""

path_spotify = "my_path_to_spotify_csv"
data = pd.read_csv(path_spotify)

def id_to_name(spotify_data, song_id):
    """

    :param spotify_data:pd>DataFrame| Spotify dataset
    :param song_id: list| List of recommended songs
    :return: List| List of song anmes
    """
    spotify_data = spotify_data[spotify_data.media_id.isin(songs)]\
        .drop_duplicates(['media_id'], keep='first')

    spotify_data = spotify_data['spotify_name']

    return list(spotify_data)
