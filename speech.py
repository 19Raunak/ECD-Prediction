import pickle
import speech_recognition as sr
import io
import pandas as pd
import liwc
import re
from collections import Counter
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def tokenize(speech_text):
    for match in re.finditer(r'\w+', speech_text, re.UNICODE):
        yield match.group(0)

def cosine_to_probability_piecewise(cosine_similarity):
    if cosine_similarity <= 0.2:
        probability = 90 + 25 * cosine_similarity
    elif 0.2 < cosine_similarity <= 0.6:
        probability = 95 - 112.5 * (cosine_similarity - 0.2)
    else:
        probability = 6 + 5 * (1 - cosine_similarity)
    return round(probability, 2)

def compute_liwc_categories(speech_text, category_names, parse):
    tokens = list(tokenize(speech_text.lower()))
    total_tokens = len(tokens)
    category_frequencies = {category: 0 for category in category_names}
    category_counts = Counter(category for token in tokens for category in parse(token))
    for category, count in category_counts.items():
        category_frequencies[category] = count / total_tokens if total_tokens > 0 else 0
    return category_frequencies

def compute_liwc_datalist(model, liwc_df):
    input_row_format = pd.read_csv('input_row_format.csv')
    liwc_df = liwc_df.reindex(columns=input_row_format.columns, fill_value=0)
    y_prob = model.predict_proba(liwc_df)
    dementia_prob = y_prob[:, 1]
    dementia_prob_rounded = (dementia_prob * 10).round().astype(int)
    return dementia_prob_rounded[0]

##def compute_cosine_similarity_datalist(text, desc_text):
    ##vectorizer = TfidfVectorizer()
    ##tfidf_matrix = vectorizer.fit_transform([text, desc_text])
    ##similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    ##return similarity[0][0]

def compute_bert_similarity(keywords, sentence):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embedding = sbert_model.encode(sentence, convert_to_tensor=True)
    keyword_embedding = sbert_model.encode(keywords, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(sentence_embedding, keyword_embedding).item()
    return similarity

##def compute_final_score(dementia_prob_rounded_final, final_bert_prob):



def main():
    pic_desc = [
        "This is a picture featuring a chaotic kitchen scene. The man is busy cutting veggies while the girls are cooking something. The dustbin is smelly and overfilled with waste. The mop and bucket are lying on the floor with water spilled over. The cat is sitting in the middle. There are many items on the table. The water in the pots in the oven is boiling. The kitchen is in complete disarray.",
        "This is a picture of a typical organized kitchen. The pans are neatly hanging on the wall. There is a fridge, oven, and chimney. The sink is kept clean with no dishes to wash. There's a small vase that adds to the aesthetics of the kitchen. The floor is made of vitrified checkered tiles, which are shiny and spick-free. Such an organized and neat place makes people happy.",
        "This picture features a mom with her two kids, a girl and a boy. The mom is busy doing the dishes with the sink overflowing with water, while the children are up to some naughty behavior. It seems both are busy stealing cookies from the shelf behind their mom's back. The boy is about to fall as the stool on which he is standing seems to topple while his sister is giggling or laughing and demands more cookies from her brother.",
        "This is a lively playground scene. All people seem so happy and cheerful, especially the children. Some are enjoying the slide while others are on the swings. A girl seems to be busy sharing something with her friend sitting on the bench, while her friend seems uninterested and more focused on eating. Two children are skipping ropes. An elder seems to have come with his baby in a stroller. One person seems to walk his dog. The two children seem thirsty, as they are quenching their thirst by drinking from the tap. Some children are playing tag. The person sitting on the bench seems to be speaking on the phone. Overall, the atmosphere seems merry."
    ]

    pic_keywords = [
        ["chaotic", "kitchen", "man", "cutting", "veggies", "girls", "cooking", "dustbin", "smelly", 
        "overfilled", "mop", "bucket", "floor", "water", "spilled", "cat", "middle", "table", "items",
        "pots", "oven", "boiling", "disarray"],

        ["organized", "kitchen", "pans", "hanging", "wall", "fridge", "oven", "chimney", "sink", 
        "clean", "no dishes", "vase", "aesthetics", "floor", "checkered", "tiles", "shiny", 
        "spick-free", "neat", "happy"],

        ["mom", "kids", "girl", "boy", "dishes", "sink", "overflowing", "water", "children", 
        "naughty", "stealing", "cookies", "shelf", "boy", "fall", "stool", "topple", "sister", 
        "giggling", "laughing", "demanding"],

        ["playground", "happy", "children", "slide", "swings", "girl", "friend", "bench", "uninterested", 
        "eating", "children", "skipping", "elder", "baby", "stroller", "dog", "walking", "thirsty", 
        "drinking", "tap", "tag", "phone", "merry"]
    ]

    
    ran_idx = random.randint(0, 3)
    print(ran_idx)
    desc_text = pic_desc[ran_idx]
    desc_text_keywords = pic_keywords[ran_idx]
    text = input("Please describe the picture: ")
    
    parse, category_names = liwc.load_token_parser('LIWC2007_English.dic')
    liwc_results = compute_liwc_categories(text, category_names, parse)
    liwc_df = pd.DataFrame([liwc_results])
    
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    dementia_prob_rounded_final = compute_liwc_datalist(model, liwc_df)
    ##final_taken_prob = []
    final_bert_prob = compute_bert_similarity(desc_text, text)
    
    final_score1, final_score_2 = dementia_prob_rounded_final, final_bert_prob
    if final_score_2 < 0.5:
        print("The description is out of context!!!")
    else:
        score = int((0.3 * (final_score1 * 2)) + (0.7 * int(final_score_2 * 10)) * 10) 
        print(f"Probability of having Dementia out of 100: {score}")

if __name__ == "__main__":
    main()