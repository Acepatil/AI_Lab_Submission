import spacy
import numpy as np
import heapq
import re
from Levenshtein import distance as levenshtein_distance

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """Preprocess the text: tokenize into sentences, normalize, and remove stopwords."""
    # Normalize text: lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    processed_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        # Lemmatize, remove stopwords, and keep only alphabetic words
        words = [token.lemma_.lower() for token in doc if token.is_alpha]
        words = [word for word in words if not nlp.vocab[word].is_stop]
        processed_sentences.append(' '.join(words))
    
    return processed_sentences

def compute_all_distances(sentences1, sentences2):
    """Compute Levenshtein distances between all pairs of sentences from two lists."""
    distances = []
    for i, sent1 in enumerate(sentences1):
        for j, sent2 in enumerate(sentences2):
            dist = levenshtein_distance(sent1, sent2)
            distances.append((i, j, dist))  # Return indices and distance
    return distances

def heuristic(i, j, len1, len2):
    """Heuristic function for A* search."""
    # Heuristic estimates the minimum cost to complete the alignment
    return abs(len1 - i) + abs(len2 - j)

def a_star_search(sentences1, sentences2):
    """Align sentences from two documents using the A* search algorithm."""
    len1, len2 = len(sentences1), len(sentences2)
    
    open_set = [(0 + heuristic(0, 0, len1, len2), 0, 0, 0)]  # (f_score, g_score, i, j)
    heapq.heapify(open_set)
    g_score = {(0, 0): 0}
    came_from = {}
    
    while open_set:
        _, cost, i, j = heapq.heappop(open_set)
        
        if i == len1 and j == len2:
            break
        
        for next_i, next_j in [(i+1, j), (i, j+1), (i+1, j+1)]:
            if next_i <= len1 and next_j <= len2:
                # Ensure valid indices and compute the distance between sentences
                if next_i < len1 and next_j < len2:
                    additional_cost = levenshtein_distance(sentences1[i], sentences2[j])
                else:
                    additional_cost = 0  # No cost if skipping past end
                new_cost = cost + additional_cost
                if (next_i, next_j) not in g_score or new_cost < g_score[(next_i, next_j)]:
                    g_score[(next_i, next_j)] = new_cost
                    f_score = new_cost + heuristic(next_i, next_j, len1, len2)
                    heapq.heappush(open_set, (f_score, new_cost, next_i, next_j))
                    came_from[(next_i, next_j)] = (i, j)
    
    alignments = []
    current = (len1, len2)
    while current in came_from:
        prev = came_from[current]
        if prev[0] < len1 and prev[1] < len2:
            alignments.append((sentences1[prev[0]], sentences2[prev[1]]))
        current = prev
    alignments.reverse()
    
    return alignments

def calculate_similarity(s1, s2):
    """Calculate the similarity between two sentences based on Levenshtein distance."""
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:  # Avoid division by zero
        return 1.0
    return 1 - (dist / max_len)

def detect_plagiarism(sentences1, sentences2):
    """Detect potential plagiarism and return all similar sentence pairs."""
    alignments = a_star_search(sentences1, sentences2)
    plagiarized_pairs = []
    
    for sent1, sent2 in alignments:
        similarity = calculate_similarity(sent1, sent2)
        plagiarized_pairs.append((sent1, sent2, similarity))
    
    return plagiarized_pairs

def calculate_plagiarism_level(plagiarized_pairs):
    """Calculate the overall level of plagiarism."""
    if not plagiarized_pairs:
        return 0.0  # No plagiarism detected
    
    total_similarity = sum(similarity for _, _, similarity in plagiarized_pairs)
    num_pairs = len(plagiarized_pairs)
    return total_similarity / num_pairs  # Average similarity score

# Example usage
text1 = "Artificial intelligence is a rapidly growing field that is transforming industries worldwide. It enables machines to learn from experience and perform tasks typically requiring human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI is also being integrated into healthcare, finance, education, and many other sectors, providing new solutions and enhancing productivity. As AI technologies continue to evolve, ethical concerns regarding privacy, bias, and job displacement must be addressed."
text2 = "The field of artificial intelligence is rapidly growing and transforming industries globally. Machines are enabled by AI to learn from experience and perform tasks that usually require human intelligence, including speech recognition, decision-making, visual perception, and translating languages. AI has also found applications in sectors like healthcare, finance, and education, enhancing productivity and providing innovative solutions. As AI technology evolves, ethical concerns such as privacy, bias, and job loss need to be addressed."
# Preprocess and tokenize the documents
processed_sent1 = preprocess_text(text1)
processed_sent2 = preprocess_text(text2)

print("Processed Sentences Doc1:", processed_sent1)
print("Processed Sentences Doc2:", processed_sent2)

# Detect plagiarism based on similarity
plagiarized_pairs = detect_plagiarism(processed_sent1, processed_sent2)
print("Plagiarized Pairs:")
for sent1, sent2, similarity in plagiarized_pairs:
    print(f"'{sent1}' is similar to '{sent2}' with similarity score of {similarity:.2f}")

# Calculate the overall level of plagiarism
plagiarism_level = calculate_plagiarism_level(plagiarized_pairs)
print(f"Overall level of plagiarism: {plagiarism_level:.2f}")
