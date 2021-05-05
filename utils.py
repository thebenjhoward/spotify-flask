import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pickle


model = None
with open('./model.pickle', 'rb') as fp:
    model = pickle.load(fp)

sp = None

class_name = "popularity"

class_bins = [
    (0,11),
    (11,float('inf'))
]
class_labels = ["0-10", "11-100"]

bins = {
    "artist_count": [
        0,
        [(1,1), (2,3), (3,float('inf'))],
        ["1", "2-3", "4+"]
    ],
    "artist_popularity": [
        1,
        [(0,20), (20,40),(40,60),(60,80),(80,float('inf'))],
        ["0-20", "20-40", "40-60", "60-80", "80-100"]
    ],
    "release_year": [
        2,
        # [(float("-inf"), 1930), (1930,1950), (1950,1960), (1960,1970), (1970,1980), (1980,1990), (1990,2000), (2000,2010), (2010,2015), (2015,float('inf'))],
        # ["Before 1930", "30s-40s", "50s", "60s", "70s", "80s", "90s", "00s", "Early 10s", "Late 10s"]
        [(float("-inf"), 1950), (1950,1970), (1970,1990), (1990,2000), (2000,2010), (2010,2015), (2015,float('inf'))],
        ["Before 1950", "50s-60s", "70s-80s", "90s", "00s", "Early 10s", "Late 10s"]
    ],
    "dancibility": [
        3,
        # [(0.0,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), (0.8,float('inf'))],
        # ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        [(0,0.6),(0.6,float('inf'))],
        ["0-0.6", "0.6-1"]
    ],
    "energy": [
        4,
        # [(0.0,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), (0.8,float('inf'))],
        # ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        [(0,0.8),(0.8,float('inf'))],
        ["0-0.8", "0.8-1"]
    ],
    "loudness": [
        5,
        [(float('-inf'), -15), (-15,-10), (-10,-5), (-5,-3),(-3,float('inf'))],
        ["very quiet", "quiet", "soft", "loud", "very loud"]
    ]
}

def bin_data(data, bns, labels, index=None):
    for j in range(len(data)):
        binned = False
        for i, bn in enumerate(bns):
            if(index is not None):
                if((data[j][index] >= bn[0] and data[j][index] < bn[1]) or (bn[0] == bn[1] and data[j][index] == bn[0])):
                    data[j][index] = labels[i]
                    binned = True
                    break
            else:
                if((data[j] >= bn[0] and data[j] < bn[1]) or (bn[0] == bn[1] and data[j] == bn[0])):
                    data[j] = labels[i]
                    binned = True
                    break
 
        if(not binned and index is None):
            raise Exception("Item out of range of bins: {}".format(data[j]))
        elif(not binned):
            raise Exception("Item out of range of bins: {}".format(data[j][index]))

def discretize_attribs(X, y):
    for _att, (i, abins, labs) in bins.items():
        #print(att)
        bin_data(X, abins, labs, index=i)
    
    bin_data(y, class_bins, class_labels)


def get_spotify_client():
    global sp

    with open("auth/client_creds.json") as creds_fp:
        creds = json.load(creds_fp)

    auth_manager = SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"])
    sp = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=10, retries=10)
    
    return sp

def load_classifier():
    pass

def lookup_song(tid):
    if(sp is None):
        get_spotify_client()
    
    song_info = sp.track(tid)
    features = sp.audio_features([tid])
    artist = sp.artist(song_info['artists'][0]['id'])

    inst = [ len(song_info['artists']), artist['popularity'], int(song_info['album']['release_date'][:4]), features[0]['danceability'], features[0]['energy'], features[0]['loudness']]
    pop = song_info['popularity']

    discretize_attribs([inst], [pop])

    return inst, pop


def classify_song(tid):

    instance, real = lookup_song(tid)
    return model.predict([instance]), real
