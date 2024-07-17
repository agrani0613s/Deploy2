import csv
import nltk
from nltk.corpus import wordnet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, ClassifierMixin
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Define all custom transformers and necessary functions

def text_to_dictionary(text):
    abbreviation_dict = {}
    lines = text.strip().split('\n')
    for line in lines:
        abbreviation, meaning = line.split('=')
        abbreviation_dict[abbreviation.strip()] = meaning.strip()
    return abbreviation_dict

text = '''
AFAIK=As Far As I Know
AFK=Away From Keyboard
ASAP=As Soon As Possible
ATK=At The Keyboard
ATM=At The Moment
A3=Anytime, Anywhere, Anyplace
BAK=Back At Keyboard
BBL=Be Back Later
BBS=Be Back Soon
BFN=Bye For Now
B4N=Bye For Now
BRB=Be Right Back
BRT=Be Right There
BTW=By The Way
B4=Before
B4N=Bye For Now
CU=See You
CUL8R=See You Later
CYA=See You
FAQ=Frequently Asked Questions
FC=Fingers Crossed
FWIW=For What Its Worth
FYI=For Your Information
GAL=Get A Life
GG=Good Game
GN=Good Night
GMTA=Great Minds Think Alike
GR8=Great!
G9=Genius
IC=I See
ICQ=I Seek you (also a chat program)
ILU=ILU: I Love You
IMHO=In My Honest/Humble Opinion
IMO=In My Opinion
IOW=In Other Words
IRL=In Real Life
KISS=Keep It Simple, Stupid
LDR=Long Distance Relationship
LMAO=Laugh My A.. Off
LOL=Laughing Out Loud
LTNS=Long Time No See
L8R=Later
MTE=My Thoughts Exactly
M8=Mate
NRN=No Reply Necessary
OIC=Oh I See
PITA=Pain In The A..
PRT=Party
PRW=Parents Are Watching
QPSA?=Que Pasa?
ROFL=Rolling On The Floor Laughing
ROFLOL=Rolling On The Floor Laughing Out Loud
ROTFLMAO=Rolling On The Floor Laughing My A.. Off
SK8=Skate
STATS=Your sex and age
ASL=Age, Sex, Location
THX=Thank You
TTFN=Ta-Ta For Now!
TTYL=Talk To You Later
U=You
U2=You Too
U4E=Yours For Ever
WB=Welcome Back
WTF=What The F...
WTG=Way To Go!
WUF=Where Are You From?
W8=Wait...
7K=Sick:-D Laugher
'''

abbreviation_dictionary = text_to_dictionary(text)

def to_low(x):
    return str(x).lower()
    
def remove_pun(x):
    return x.translate(str.maketrans('', '', string.punctuation))

def chat_con(text):
    new_text = []
    for w in text.split():
        if w.upper() in abbreviation_dictionary:
            new_text.append(abbreviation_dictionary[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def tokeizer(x):
    return word_tokenize(str(x))

class RemovePunTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_pun):
        self.remove_pun = remove_pun
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return [self.remove_pun(w) for w in x]

class RemoveAbbreviationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, chat_con):
        self.chat_con = chat_con
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return [self.chat_con(w) for w in x]

class RemoveStopwordsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stopwords):
        self.remove_stopwords = remove_stopwords
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return [self.remove_stopwords(w) for w in x]

class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return [self.lemmatize_sentence(sentence) for sentence in X]
    def lemmatize_sentence(self, sentence):
        words = word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        lemmatized_words = [self.lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
        return ' '.join(lemmatized_words)

class TokenizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tokeizer):
        self.tokeizer = tokeizer
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return [self.tokeizer(w) for w in x]
    
class LowercaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, to_low):
        self.to_low = to_low
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return [self.to_low(w) for w in x]

class EmotionAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, csv_file, threshold=0.1):
        self.csv_file = csv_file
        self.threshold = threshold
        self.emotion_dict, self.emotions = self._load_emotion_mappings()
    def _load_emotion_mappings(self):
        emotion_dict = {}
        with open(self.csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            emotions = [field for field in reader.fieldnames if field != 'English Word']
            for row in reader:
                word = row['English Word'].lower()
                emotion_dict[word] = {emotion: int(row[emotion]) for emotion in emotions}
        return emotion_dict, emotions
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [self._analyze_tokens(tokens) for tokens in X]
    def _analyze_tokens(self, tokens):
        total_scores = defaultdict(int)
        word_count = 0
        for token in tokens:
            if token.lower() in self.emotion_dict:
                word_count += 1
                for emotion in self.emotions:
                    total_scores[emotion] += self.emotion_dict[token.lower()][emotion]
        if word_count > 0:
            avg_scores = {emotion: total_scores[emotion] / word_count for emotion in self.emotions}
        else:
            avg_scores = {emotion: 0 for emotion in self.emotions}
        emotion_vector = [1 if avg_scores[emotion] > self.threshold else 0 for emotion in self.emotions]
        return emotion_vector

class SequentialPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, model1, model2, pivot_matrix, n_neighbors=6):
        self.model1 = model1
        self.model2 = model2
        self.pivot_matrix = pivot_matrix
        self.n_neighbors = n_neighbors
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        # Get predictions from model1 (SVM)
        svm_pred = self.model1.predict(X)
        # Convert SVM predictions to integers
        svm_pred_int = np.array([int(pred) for pred in svm_pred])
        # Ensure that the indices are within the bounds of the pivot_matrix
        svm_pred_int = np.clip(svm_pred_int, 0, len(self.pivot_matrix) - 1)
        # Use the indices to get the corresponding rows from the recommendation matrix
        knn_input = self.pivot_matrix.iloc[svm_pred_int].values
        # Use the mapped SVM predictions to find nearest neighbors
        distances, indices = self.model2.kneighbors(knn_input, n_neighbors=self.n_neighbors)
        # Return the URLs of the nearest neighbors
        nearest_urls = [self.pivot_matrix.index[idx] for idx in indices.flatten()]
        return nearest_urls

# Load the recommendation system from the pickle file
recommendation_system = joblib.load("model.joblib")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def sensenlpmodel():
    if request.method == 'POST':
        try:
            inputstring = request.form.get('inputstring')
            if inputstring is not None:
                input_list = [inputstring]
                video_urls = recommendation_system.predict(input_list)
                return jsonify({'Recommendations': video_urls})
            else:
                return jsonify({'error': 'No input string provided'})
        except Exception as e:
            return jsonify({'error': str(e)})
    return "Please Enter POST request"

if __name__ == '__main__':
    app.run(debug=True)
