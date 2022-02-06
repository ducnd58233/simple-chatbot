from utils.nltk_utils import stem, tokenize

def preprocess_data(intents):
    all_words = []
    tags = []
    answer_token = []
    
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        
        for pattern in intent['patterns']:
            word = tokenize(pattern)
            all_words.extend(word)
            answer_token.append((word, tag))
    
    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    
    return all_words, tags, answer_token