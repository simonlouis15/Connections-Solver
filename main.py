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


model = api.load("word2vec-google-news-300")
sentences_model  = SentenceTransformer('all-mpnet-base-v2')


def get_ngram_data(ngrams, start_year=2018, end_year=2019, corpus=26, smoothing=3):
    base_url = "https://books.google.com/ngrams/json"
    ngrams_str = ','.join(ngrams)

    params = {
        'content': ngrams_str,
        'year_start': start_year,
        'year_end': end_year,
        'corpus': corpus,  # Corpus 26 is English
        'smoothing': smoothing,
        'direct_url': 't1'
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

# List of words to group
words_list = [
    'bash', 'bag', 'color', 'sack',
    'reception', 'blowout', 'land', 'pan',
    'cut', 'blast', 'yard', 'score',
    'slam', 'snag', 'trim', 'attempt'

]

#------------------------------------------------CONNECTIONS WORD LIST-----------------------------------------------------------------------------------------------------------------

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

class Group:
    def __init__(self, index, variance, words):
        self.index = index
        self.variance = variance
        self.words = words


#--------------------------------------------------HELPER FUNCTIONS----------------------------------------------------------------------------------------------------------------------

variances = []
def get_even_clusters(X, cluster_size, num_clusters):

    kmeans = KMeansConstrained(num_clusters, n_init=20, random_state=0)
    kmeans.fit(X) #assign each word vector to one of 4 clusters

    centers = kmeans.cluster_centers_ #find the centroid of each cluster
    centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1]) 
    distance_matrix = cdist(X, centers)

    clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size

    # Compute within-cluster variance
    for cluster_num in range(num_clusters):
        cluster_points = X[clusters == cluster_num]
        distances = cdist(cluster_points, [centers[cluster_num]], 'euclidean')

        pairwise_distances = squareform(pdist(cluster_points))

        np.fill_diagonal(pairwise_distances, np.inf)

        # Find the distance to the closest point for each point in the cluster
        min_distances = np.min(pairwise_distances, axis=1)

        # Find the index of the point with the maximum of these closest distances
        farthest_point_index = np.argmax(min_distances)
        farthest_point = cluster_points[farthest_point_index]

        #new_data_set = X.remove(farthest_point)
                
        variance = np.var(distances)
        variances.append(variance)

    # Print variances for each cluster
    for i, variance in enumerate(variances):
        print(f"Cluster {i+1} Within-Cluster Variance: {variance:.3f}")
    
    return clusters

def get_best_group(X, num_clusters=1):
    kmeans = KMeans(num_clusters, n_init=20, random_state=0).fit(X) #assign each word vector to one of 4 clusters

    distances = kmeans.transform(X).flatten()
    closest_indices = np.argsort(distances)[:4]
    
    return closest_indices

def find_suffix():
    vocab = model.key_to_index

    master_token_list = []
    for word in words_list: #for each word in the connections
        #get all the phrases that contain that word
        matching_tokens = [phrase.lower() for phrase in vocab if phrase.lower().endswith(word.lower())]
        #print(matching_tokens)
        #now check if the next word contains the same suffix
        cleaned_tokens = []
        for token in matching_tokens:
            if token.count('_') > 1: #contains more than two words
                continue #dont add this token
            elif token.count('_') == 1 and token.endswith('_' + word): #common phrase
                cleaned_token = token.replace('_', '')
                #cleaned_tokens.append(cleaned_token[len(word):])
                cleaned_tokens.append(cleaned_token[:(len(cleaned_token)-len(word))])
            else: #homophone
                #cleaned_tokens.append(token[len(word):])
                cleaned_tokens.append(token[:(len(token)-len(word))])

        master_token_list.append(cleaned_tokens)
    

    intersection_matrix = []
    best_score = -1
    best_group = []
    words = []
    possible_other_groups = []
    for i in range(len(words_list)):
        for j in range(i+1, len(words_list)):
            for k in range(j+1, len(words_list)):
                for l in range (k+1, len(words_list)):
                    a_set = set(master_token_list[i])
                    b_set = set(master_token_list[j])
                    c_set = set(master_token_list[k])
                    d_set = set(master_token_list[l])

                    if (a_set & b_set & c_set & d_set):
                        
                        intersection = list(a_set & b_set & c_set & d_set)

                        condition = lambda word: len(word) > 2

                        condition2 = lambda word: any(not letter.isalnum() for letter in word)

                        filtered_intersections = [word for word in intersection if condition(word) and not condition2(word)]

                        nono_list = ['ing', 's', 'y', 'ers', 'man', 'the', 'this', 'new']
                        if len(filtered_intersections) == 1 and filtered_intersections[0] not in nono_list: #if only one word intersected
                            words = [words_list[i], words_list[j], words_list[k], words_list[l]]
                            intersection_matrix.append((filtered_intersections[0], words))
                        elif len(filtered_intersections) > 1 and filtered_intersections[0] not in nono_list:
                            if len(words_list) == 8:
                                words = [words_list[i], words_list[j], words_list[k], words_list[l]]
                                other_group = [word for word in words_list if word not in words]
                                score, matrix = compute_confidence(other_group)
                                if score > best_score:
                                    best_score = score
                                    best_group = other_group
                            else:
                                words = [words_list[i], words_list[j], words_list[k], words_list[l]]
                                other_words = [word for word in words_list if word not in words]
                                other_group = find_strongest_group(other_words)
                                score, matrix = compute_confidence(other_group)
                                if score > best_score:
                                    best_score = score
                                    best_group = other_group
                                
    if best_group:
        "be wary of this"
        return best_group

    filtered_intersection_matrix = []
    if len(intersection_matrix) > 1:

        average_freqs = []
        frequencies = []
        for inter in intersection_matrix:

            full_words = []
            for word in inter[1]:
                if (inter[0] + word) in model.index_to_key:
                    #common start
                    full_words.append(inter[0] + word)
                else:
                    full_words.append(inter[0] + ' ' + word)

            """
            for word in inter[1]:
                if (word + inter[0]) in model.index_to_key:
                    #common start
                    full_words.append(word + inter[0])
                else:
                    full_words.append(word + ' ' + inter[0])
            """
            print(full_words)
            
            data = get_ngram_data(full_words)
            #common end
            #data = get_ngram_data([word + inter[0] for word in inter[1]])
            frequencies = [entry['timeseries'][-1] for entry in data]
            avg_freq = sum(frequencies) / 4
            #print([entry['timeseries'] for entry in data])
            if len(frequencies) < 4:
                average_freqs.append(-1)
            else:    
                average_freqs.append(avg_freq)

        highest_freq_index = average_freqs.index(max(average_freqs))
        filtered_intersection_matrix = intersection_matrix[highest_freq_index]

        return filtered_intersection_matrix
  
    else:
        return intersection_matrix

#---------------------------------------GET WORD VECTORS FOR CONNECTIONS WORDS----------------------------------------------------------------------


#-----------------------------------------------GROUP INTO CLUSTERS---------------------------------------------------------------------------------------------------------------------------
def solver():
    word_vectors = []
    for word in words_list:
        word_vectors.append(model[word])

    word_vectors_2d = np.array(word_vectors)
    
    num_clusters= math.ceil((len(words_list) / 4))
    balanced_clusters = get_even_clusters(word_vectors_2d, 4, num_clusters)
    print(f"best bet is group: {variances.index(min(variances)) + 1}")


    grouped_words = [[] for elem in range (4)] #creates 4 empty lists
    for i, label in enumerate(balanced_clusters): #for each word in the cluster (label is between 0-3)
        grouped_words[label].append(words_list[i]) #add the word to the list of words in the cluster

    print(grouped_words)
    return grouped_words

wrong_combinations = []
def error_handling(grouped_words):
    error_count = 0
    best_group = grouped_words[variances.index(min(variances))]
    best_group.sort()
    print("original: ", best_group)

    for answer in answer_key:
        answer.sort()
        print(answer)

    while 1:
        if best_group in answer_key: #im not sure if order matters (prob does)
            print("we got it right: ", best_group)
            variances.clear()
            grouped_words.clear()
            for word in best_group:
                words_list.remove(word)
            grouped_words = solver()
            best_group = grouped_words[variances.index(min(variances))].sort()
        else:
            print("mistake!")
            error_count += 1
            if error_count == 4:
                break
            grouped_words.remove(grouped_words[variances.index(min(variances))])
            variances.remove(min(variances))
            best_group = grouped_words[variances.index(min(variances))]
            best_group.sort()
            print("new: ", best_group)



#indices = get_best_group(words_list)
#print([words_list[index] for index in indices])

def group_by_context():
    # Calculate the aggregate similarity for each combination
    best_similarity = -1
    best_combination = None

    for group in combinations(words_list, 4):

        
        word_vectors = []
        for word in group:
            word_vectors.append(model[word])
        
        word_vectors_2d = np.array(word_vectors)

        # Calculate pairwise similarities
        pairwise_similarities = [
            cosine_similarity([word_vectors_2d[i]], [word_vectors_2d[j]])[0][0]
            for i in range(4) for j in range(i + 1, 4)
        ]
    
        # Aggregate similarity (e.g., by averaging)
        avg_similarity = np.mean(pairwise_similarities)

        
        # Track the best combination
        if avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_combination = group
            print(best_combination, ": ", best_similarity)
            

    return best_similarity




"""
def check_overlap():
    grouped_words = solver()

    for group in grouped_words:
        group.sort()

    best_group = find_strongest_group(words_list).sort()

    for group in grouped_words:
        intersection = best_group & group

        if len(intersection) == 3: #three word match (pretty close)
            #here we remove word with lowest cosine similarity to the rest of the words
            # and replace it with the different word from the intersected group
"""

#error_handling(grouped_words)
def starts_with_year(line):
    match = re.match(r"^\d{4}\b", line)
    return bool(match)

def get_word_data(word):
    #print("not again :()")
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
    
    response = requests.get(url, params=params)
    data = response.json()
    
    pages = data.get("query", {}).get("pages", {})
    
    for page_id, page_data in pages.items():
        if "extract" in page_data:
            return page_data["extract"]
    
    return None

def should_start_collecting(line):
    for section in ["noun", "verb", "adjective"]:
        if line.lower().startswith(section):
            return True
        
    return False

def should_skip_line(line):
    sections = [
         "troponym", "antonym", "homonym", "polynym", "homophone",
        "plural", "vulgar", "ipa(key)", "obsolete", "archaic", " iso ", "sex", 
        "alternative spelling", "penis"
    ]
    processed_line = line.lower()
    #print("processed line: ", processed_line)

    res = [word for word in sections if (word in processed_line)]

    if bool(res) or not processed_line.isascii():
        #print("bad line: ", line)
        return True
    
def parse_definitions(content, word):
    if content:
        lines = content.split('\n')
    else:
        return
    
    definitions = []
    possible_definition_types = 0
    start_collecting = False
    continueAgain = False

    for index, line in enumerate(lines):
        line = line.strip()

        #just done each time we enter a new definition section to get rid of useless def
        if continueAgain ==  True:
            continueAgain = False
            continue

        #print(line)
        if should_start_collecting(line):
            possible_definition_types += 1
            continueAgain = True
            if possible_definition_types == 4:
                break

            start_collecting = True
            #print("start collecting is: ", str(start_collecting))
            continue

        if start_collecting:
            #new section is starting
            if not lines[index] and not lines[index+1]:
                start_collecting = False
                #print("start collecting is: ", str(start_collecting))
                continue

            if should_skip_line(line):
                continue

            if line.startswith("c. ") or line.startswith(',') or starts_with_year(line):
                continueAgain = True
                continue

            # Process exisiting lines
            if line and line != word:
                #print("good line: ", line)
                definitions.append(line)
                if len(definitions) == 15: #hard cut-off
                    break


    return definitions

def get_word_info(word):
    word_data = get_word_data(word)
    definitions = parse_definitions(word_data, word)
    if definitions:
        return definitions
    else:
        return '0'

def wordnet_definitions(word):
    results = set()
    words = [word, word + 's']
    

    # Get synsets for each word
    word_synsets = [wn.synsets(elem) for elem in words]

    for variation in word_synsets:
        for syn in variation:
            definition = syn.definition()
            examples = syn.examples()
            synonyms = [lemma.name() for lemma in syn.lemmas()]
            hypernyms = [hypernym.name().split('.')[0] for hypernym in syn.hypernyms()]
            hyponyms = [hyponym.name().split('.')[0] for hyponym in syn.hyponyms()]

            # Append examples to the definition
            if examples:
                definition += ". Examples: " + "; ".join(examples)
            if synonyms:
                definition += ". Synonyms: " + "; ".join(synonyms)
            if hypernyms:
                definition += ". Hypernyms: " + "; ".join(hypernyms)
            if hyponyms:
                definition += ". Hyponyms: " + "; ".join(hyponyms)

            results.add(definition)
        
    return results

def get_definitions(words):
    
    master_def_dictionary = {}
    for word in words: #for each of the 16 words
        definition_vectors = []
        #definitions = get_word_info(word) #get all the definitions associated with that word
        #definitions = get_word_info(word)
        #if len(definitions) < 2:
        definitions = wordnet_definitions(word)
            
        for defin in definitions: #now for each of those definitions
            embedding = sentences_model.encode(defin, convert_to_tensor=True) #get the vector embedding
            definition_vectors.append(embedding)

        master_def_dictionary[word] = definition_vectors

    return master_def_dictionary

def compare_definitions(embedding1, embedding2):

    # Compute Cosine Similarity
    cosine_sim = util.cos_sim(embedding1, embedding2)

    return cosine_sim.item()

definition_dict = get_definitions(words_list)
def compute_confidence(group):
    definitions_pairwise_matrix = []

    score = 0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            
            definitions1 = definition_dict[group[i]]
            definitions2 = definition_dict[group[j]]
            best_score = -1
            best_def1 = ''
            best_def2 = ''
            for defin1 in definitions1:
                for defin2 in definitions2:
                    similarity_score = compare_definitions(defin1, defin2)
                    if similarity_score > best_score:
                        best_score = similarity_score
                        best_def1 = defin1
                        best_def2 = defin2
            
            definitions_pairwise_matrix.append([group[i], group[j], best_def1, best_def2])
            score += best_score
    return score / 6, definitions_pairwise_matrix  # There are 6 pairs in a group of 4



def find_strongest_group(words):
    best_group = None
    best_score = float('-inf')
    
    for group in combinations(words, 4):

        repeated_words = False
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                if group[i].lower() in group[j].lower():
                    repeated_words = True

        if repeated_words:
            #print(group)
            continue

        score, matrix = compute_confidence(group)

        if score > best_score:
            best_score = score
            best_group = group
            print(best_group, "has score of: ", best_score)
    
    return best_group, best_score

def combinations_grouper():
    method1_group, score = find_strongest_group(words_list)
    method2_groups = solver()

    most_intersections = -1
    comparison_group = []
    intersection = []
    for group in method2_groups:
        num_intersections = len(list(set(method1_group) & set(group)))
        if num_intersections > most_intersections and num_intersections > 1:
            intersection = list(set(method1_group) & set(group))
            most_intersections = num_intersections
            comparison_group = group
    
    undetermined_words = list(set(method1_group) ^ set(comparison_group))

    for group in combinations(undetermined_words, 2):
        group = list(group)
        print(group)
        for word in intersection:
            group.append(word)
        score, matrix = compute_confidence(group)
        print(group, ": ", score)

def find_wrong_word(group):

    best_score, matrix = compute_confidence(group)
    word_score = [0, 0, 0, 0]
    for word_info in matrix:
        if word_info[0] == group[0] or word_info[1] == group[0]:
            score = compare_definitions(word_info[2], word_info[3])
            #print(word_info[0], "and ", word_info[1], " have a score of: ", score)
            word_score[0] += score

        if word_info[0] == group[1] or word_info[1] == group[1]:
            score = compare_definitions(word_info[2], word_info[3])
            #print(word_info[0], "and ", word_info[1], " have a score of: ", score)
            word_score[1] += score

        if word_info[0] == group[2] or word_info[1] == group[2]:
            score = compare_definitions(word_info[2], word_info[3])
            #print(word_info[0], "and ", word_info[1], " have a score of: ", score)
            word_score[2] += score

        if word_info[0] == group[3] or word_info[1] == group[3]:
            score = compare_definitions(word_info[2], word_info[3])
            #print(word_info[0], "and ", word_info[1], " have a score of: ", score)
            word_score[3] += score

    sucky_word = group[word_score.index(min(word_score))]

    print("the sucky word is: ", sucky_word)

#find_wrong_word(['buff', 'cream', 'fawn', 'tan'])

def homophones():
    df = pd.read_csv("The-Big-List-of-Homophones.csv", usecols=['1', '2', '3', '4', '5', '6'], encoding='latin-1')
    df_flat = df.values.flatten()

    homophone_tuples = [tuple.split('/') for tuple in df_flat if type(tuple) == str]

    homophones = []
    for tuple in homophone_tuples:
        #print(tuple)
        for word in words_list:
            if word in tuple:
                homophones.append(tuple)

    real_words = []
    for pair in homophones:
        for word in pair:
            if word not in words_list and word not in real_words:
                real_words.append(word)

    print(real_words)
    additional_definitions = get_definitions(real_words)
    definition_dict.update(additional_definitions)

    best_group = find_strongest_group(real_words)

    return best_group
      

#combinations_grouper()
#find_strongest_group(words_list)
#group_by_context()

#print(find_suffix())
#grouped_words = solver()

#score, matrix = compute_confidence(['job', 'position', 'post', 'station'])
#print(score)

#phrases = [phrase for phrase in model.index_to_key if 'head' in phrase.lower()]
#print(f"\n\n {phrases}")


"""
score, matrix = compute_confidence(['brush', 'cape', 'clippers', 'gel',])
score1, matrix = compute_confidence(['fire', 'heat', 'kick', 'spice',])
score2, matrix = compute_confidence(['tarot', 'set', 'magic', 'baseball'])
print(score)
print(score1)
print(score2)
avg = (score + score1 + score2) / 3
print(avg)

best1, bs1 = find_strongest_group(words_list)
for word in best1:
    if word in words_list:
        words_list.remove(word)

definition_dict = get_definitions(words_list)

best2, bs2 = find_strongest_group(words_list)
for word in best2:
    if word in words_list:
        words_list.remove(word)
definition_dict = get_definitions(words_list)

best3, bs3 = find_strongest_group(words_list)

best_avg = (bs1 + bs2 + bs3) / 3
print(best_avg)
"""




for word in ['blowout', 'blow_out', 'blow-out']:
    print("the definitions of ", word, " are: ")

    #definitions = get_word_info(word)
    #if len(definitions) < 2:
    definitions = wordnet_definitions(word)

    for i, defin in enumerate(definitions):
        print("definition ", i, ": ", defin)
    print("\n")


#print(get_word_data("office"))



#print(get_word_data('cream'))
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------


#print(homophones())



