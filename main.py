#first import dependencies
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import linear_sum_assignment
import gensim.downloader as api
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.collocations import *
import numpy as np
import pandas as pd
import math
import requests
from collections import Counter
from itertools import combinations, product
from sentence_transformers import SentenceTransformer, util
import re


print("The connections solver has now begun. It will output a series of combinations of 4 words in increasing similarity along with their confidence scores out of 1")
print("Please be patient as it loads the model and analyzes the words.")
# Load the pre-trained word2vec model (Google News 300-dimensional word vectors)
model = api.load("word2vec-google-news-300")

# Load the pre-trained SentenceTransformer model (MPNet-based embedding for sentences)
sentences_model = SentenceTransformer('all-mpnet-base-v2')


# Function to retrieve ngram data from Google's Ngram Viewer
def get_ngram_data(ngrams, start_year=2018, end_year=2019, corpus=26, smoothing=3):
    # Base URL for the Ngram Viewer API
    base_url = "https://books.google.com/ngrams/json"
    
    # Join the ngram list into a comma-separated string
    ngrams_str = ','.join(ngrams)

    # Set the parameters for the API request
    params = {
        'content': ngrams_str,  # The ngrams we want data for
        'year_start': start_year,  # The start year for the data
        'year_end': end_year,  # The end year for the data
        'corpus': corpus,  # Corpus 26 represents the English language
        'smoothing': smoothing,  # Apply smoothing to the frequency data
        'direct_url': 't1'  # This parameter configures the response to return in a direct JSON format
    }
    
    # Make the request to the Ngram Viewer API
    response = requests.get(base_url, params=params)
    
    # If the request is successful (status code 200), return the JSON response
    if response.status_code == 200:
        return response.json()
    # If the request fails, raise an exception with the error status code
    else:
        response.raise_for_status()

# List of words to group
words_list = [
    'raven', 'silo', 'hitch', 'promo',
    'trailer', 'discog', 'fish', 'tractor',
    'bally', 'farm', 'cloister', 'gather',
    'seperate', 'hunt', 'axles', 'seclude'
]

#------------------------------------------------PAST CONNECTIONS WORD LISTS-----------------------------------------------------------------------------------------------------------------

#'Babe', 'buzz', 'booboo', 'daisy',
#'jasmine', 'Mickey', 'dance', 'Yogi'
#'goof', 'sting', 'petunia', 'flub', 
#'poppy', 'Lou', 'gaffe', 'pollinate'

#'strawberry', 'milk', 'office', 'run',
#     'blonde', 'rose', 'work', 'teams',
#    'jobs', 'function', 'edge', 'mars',
#    'windows', 'vice', 'devil', 'operate',

#'bowie', 'head', 'full', 'butter',
#'lead', 'throne', 'mercury', 'king',
#'queen', 'butterfly', 'john', 'gold',
#'butcher', 'twin', 'tin', 'can'

#"marked", "said", "spoke", 
#"clear", "handle", "pronounced",
#"designation", "striking", "moniker",
# "told", "sobriquet", "voiced"

#'match', 'phone', 'bud', 'range',
# 'sierra', 'dirt', 'mate', 'pal',
# 'reach', 'check', 'natty', 'scope',
# 'complement', 'stella', 'extent', 'partner',

#'ok', 'cowboy', 'ankle', 'sun',
#'tide', 'yoyo', 'yoyo', 'ma',
#'fair', 'gogo', 'average', 'elevator',
#'Soso', 'high', 'or', 'thigh-high'

#'nerd', 'grump', 'snooze', 'WHAT_IF',
#'say', 'do', 'dope', 'runt', 
#'sleep', 'kiss', 'suppose', 'alarm',
#'hour', 'perhaps', 'time', 'whopper'

#'cat', 'crank', 'blow', 'pop',
#'lion', 'grinch', 'wind', 'turtle',
#'crab', 'reel', 'draft', 'ram',
#'puff', 'bull', 'turn', 'gust'

#'balloon', 'bar', 'cake', 'fizz',
#'pie', 'bubble', 'tire', 'coat',
#'sling', 'smear', 'punch', 'line',
#'floatie', 'sour', 'plaster', 'basketball'

#'egg', 'story', 'sun', 'screen', 
#'moon', 'reel', 'streak', 'post',
#'globe', 'deck', 'speaker', 'floor',
#'toilet_paper', 'mirror', 'level', 'projector'

#'tie', 'tan', 'fan', 'fawn', 
#'check', 'finish', 'monitor', 'buff',
#'boa', 'screen', 'cream', 'bask',
#'terminal', 'corset', 'display', 'gloves'

#'national', 'talk', 'slip', 'latin',
#'jargon', 'steal', 'budget', 'tiptoe',
#'whispers', 'environ', 'plead', 'enterprise',
#'creep', 'speculation', 'thrifty', 'rumbling'

#'slice', 'sneak', 'tire', 'support',
#'bagel', 'slip', 'loaf', 'whiff',
#'blessing', 'wade', 'lifesaver', 'consent',
#'wreath', 'hook', 'approval', 'shank'

#['kick', 'thunder', 'brush', 'heat'],
#['gel', 'set', 'magic', 'fire'],
#['clippers', 'roar', 'cape', 'tarot'],
#['boom', 'spice', 'baseball', 'crash']

#'martini', 'teeth', 'humor', 'pendulum',
#'steps', 'tea_bag', 'tetherball', 
#'teetotaler', 'boomer', 'desert', 
# 'yo-yo', 'teeter-totter', 'blues' 

#'player', 'gamble', 'brown', 'young',
#'count', 'smith', 'noble', 'duke',
#'upright', 'consider', 'grand', 'judge',
# 'electronic', 'regard', 'howard', 'johnson' 

#'record', 'weird',
#'stamp', 'station', 'comic',
#'funny', 'coin', 'post', 
#'off', 'position', 'curious', 'job',

#'too', 'real', 'fore', 'spare',
#'won', 'arrow', 'extra', 'pound',
#'knuckle', 'excuse', 'over', 'block',
#'save', 'beyond', 'pardon', 'yen'

#'cater', 'bowl', 'throw', 'star',
#'caret', 'crate', 'hash',
# 'bed', 'brace',
#'plan', 'collar', 'host'

#Word groupings class indicating the similarity of groups of 4 words
class Group:
    def __init__(self, index, variance, words):
        self.index = index
        self.variance = variance
        self.words = words

#-----------------------------------------------GROUP INTO CLUSTERS---------------------------------------------------------------------------------------------------------------------------
# Function to check if a line starts with a year (4 digits)
def starts_with_year(line):
    match = re.match(r"^\d{4}\b", line)
    return bool(match)

# Function to get word data from Wiktionary API
def get_word_data(word):
    url = "https://en.wiktionary.org/w/api.php"
    
    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": word,
        "prop": "extracts",
        "explaintext" : True,
        "exsectionformat": "plain",
    }
    
    response = requests.get(url, params=params)  # Send the API request
    data = response.json()  # Parse the response data
    
    pages = data.get("query", {}).get("pages", {})  # Get the pages from the response data
    
    # If the page data contains an extract, return the extract
    for page_id, page_data in pages.items():
        if "extract" in page_data:
            return page_data["extract"]
    
    return None  # Return None if no extract is found

# Function to determine if a line should start collecting (for noun, verb, adjective sections)
def should_start_collecting(line):
    for section in ["noun", "verb", "adjective"]:
        if line.lower().startswith(section):  # Check if the line starts with a section
            return True
    return False

# Function to check if a line should be skipped based on certain conditions
def should_skip_line(line):
    sections = [
         "troponym", "antonym", "homonym", "polynym", "homophone",
        "plural", "vulgar", "ipa(key)", "obsolete", "archaic", " iso ",
        "alternative spelling",
    ]
    processed_line = line.lower()  # Process the line to lowercase

    # Check if any of the sections appear in the processed line
    res = [word for word in sections if (word in processed_line)]

    # If the line contains certain keywords or is non-ASCII, it should be skipped
    if bool(res) or not processed_line.isascii():
        return True
    
# Function to parse definitions from the content of a word's page
def parse_definitions(content, word):
    if content:
        lines = content.split('\n')  # Split content into lines
    else:
        return
    
    definitions = []  # List to store definitions
    possible_definition_types = 0  # Counter for different types of definitions
    start_collecting = False  # Flag to indicate when to start collecting definitions
    continueAgain = False  # Flag to continue after a section is done

    for index, line in enumerate(lines):
        line = line.strip()

        # Skip lines if continueAgain is True
        if continueAgain ==  True:
            continueAgain = False
            continue

        # If the line starts with a section like "noun", "verb", "adjective", start collecting
        if should_start_collecting(line):
            possible_definition_types += 1
            continueAgain = True
            if possible_definition_types == 4:
                break

            start_collecting = True
            continue

        if start_collecting:
            # New section is starting, stop collecting if lines are empty
            if not lines[index] and not lines[index+1]:
                start_collecting = False
                continue

            # Skip lines based on certain conditions
            if should_skip_line(line):
                continue

            # Skip lines that start with "c.", commas, or years
            if line.startswith("c. ") or line.startswith(',') or starts_with_year(line):
                continueAgain = True
                continue

            # Process valid lines (add to definitions)
            if line and line != word:
                definitions.append(line)
                if len(definitions) == 15:  # Limit the number of definitions to 15
                    break

    return definitions

# Function to get word information (data + definitions)
def get_word_info(word):
    word_data = get_word_data(word)  # Get word data from Wiktionary API
    definitions = parse_definitions(word_data, word)  # Parse the definitions from the data
    if definitions:
        return definitions
    else:
        return '0'  # Return '0' if no definitions are found

# Function to get WordNet definitions for a word and its plural form
def wordnet_definitions(word):
    results = set()  # Set to store unique results
    words = [word, word + 's']  # Include singular and plural forms
    
    # Get synsets for both forms of the word
    word_synsets = [wn.synsets(elem) for elem in words]

    # Iterate over the synsets to get definitions, examples, synonyms, hypernyms, and hyponyms
    for variation in word_synsets:
        for syn in variation:
            definition = syn.definition()
            examples = syn.examples()
            synonyms = [lemma.name() for lemma in syn.lemmas()]
            hypernyms = [hypernym.name().split('.')[0] for hypernym in syn.hypernyms()]
            hyponyms = [hyponym.name().split('.')[0] for hyponym in syn.hyponyms()]

            # Append examples, synonyms, hypernyms, and hyponyms to the definition
            if examples:
                definition += ". Examples: " + "; ".join(examples)
            if synonyms:
                definition += ". Synonyms: " + "; ".join(synonyms)
            if hypernyms:
                definition += ". Hypernyms: " + "; ".join(hypernyms)
            if hyponyms:
                definition += ". Hyponyms: " + "; ".join(hyponyms)

            results.add(definition)  # Add the definition to the results set
        
    return results

# Function to get definitions for a list of words
def get_definitions(words):
    master_def_dictionary = {}  # Dictionary to store word definitions

    # For each word in the list of words
    for word in words:
        definition_vectors = []  # List to store the word's definition vectors
        definitions = wordnet_definitions(word)  # Get WordNet definitions for the word
            
        # For each definition, generate a vector embedding
        for defin in definitions:
            embedding = sentences_model.encode(defin, convert_to_tensor=True)  # Get the vector embedding
            definition_vectors.append(embedding)

        # Add the word and its corresponding definition vectors to the dictionary
        master_def_dictionary[word] = definition_vectors

    return master_def_dictionary

# Function to compare the similarity of two definition embeddings
def compare_definitions(embedding1, embedding2):
    cosine_sim = util.cos_sim(embedding1, embedding2)  # Compute Cosine Similarity
    return cosine_sim.item()  # Return the similarity score

definition_dict = get_definitions(words_list)
# Function to compute confidence score for a group of words based on their definitions
def compute_confidence(group):
    definitions_pairwise_matrix = []  # List to store pairwise definition comparisons
    score = 0  # Variable to store the cumulative score

    # For each pair of words in the group
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            definitions1 = definition_dict[group[i]]  # Get definitions for word 1
            definitions2 = definition_dict[group[j]]  # Get definitions for word 2
            best_score = -1  # Initialize the best score
            best_def1 = ''  # Best definition for word 1
            best_def2 = ''  # Best definition for word 2

            # For each definition of word 1 and word 2, compare the similarity
            for defin1 in definitions1:
                for defin2 in definitions2:
                    similarity_score = compare_definitions(defin1, defin2)  # Get similarity score
                    if similarity_score > best_score:
                        best_score = similarity_score
                        best_def1 = defin1
                        best_def2 = defin2
            
            definitions_pairwise_matrix.append([group[i], group[j], best_def1, best_def2])  # Store the pairwise comparison
            score += best_score  # Add the best score to the cumulative score

    return score / 6, definitions_pairwise_matrix  # Return the average score for the group (there are 6 pairs in a group of 4)

# Function to find the strongest group of words based on definition similarity
def find_strongest_group(words):
    best_group = None  # Variable to store the best group
    best_score = float('-inf')  # Initialize the best score to negative infinity
    
    # Generate all combinations of 4 words from the list
    for group in combinations(words, 4):
        repeated_words = False  # Flag to indicate if there are repeated words in the group

        # Check for repeated words in the group (case-insensitive)
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                if group[i].lower() in group[j].lower():
                    repeated_words = True
                    continue

        if repeated_words:
            continue  # Skip this group if it contains repeated words

        score, matrix = compute_confidence(group)  # Compute the confidence score for the group

        # Update the best group and best score if necessary
        if score > best_score:
            best_score = score
            best_group = group
            print(best_group, "has score of: ", best_score)
    
    return best_group, best_score

# Function to find homophones from a CSV file and update definitions
def homophones():
    df = pd.read_csv("The-Big-List-of-Homophones.csv", usecols=['1', '2', '3', '4', '5', '6'], encoding='latin-1')
    df_flat = df.values.flatten()  # Flatten the dataframe

    # Convert homophone pairs into tuples
    homophone_tuples = [tuple.split('/') for tuple in df_flat if type(tuple) == str]

    homophones = []  # List to store homophones

    # For each homophone pair, check if the words exist in the list and add to homophones list
    for pair in homophone_tuples:
        for word in pair:
            if word in words_list:
                homophones.append(word)
    
    # Remove duplicate homophones from the list
    homophones = list(set(homophones))

    return homophones  # Return the list of homophones


      
#UNCOMMENT TO TRY OUT THE SOLVER
find_strongest_group(words_list)
print("The connections solver is now complete. Thanks for checking out my project!")
#print(homophones())


