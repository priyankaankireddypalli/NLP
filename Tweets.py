import pandas as pd
Tweets_data = pd.read_csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\NLPTopicModelling\\Data.csv')
Tweets_data.head(10)
Tweets_data.columns
import re
HANDLE = '@\w+'
LINK = 'https?://t\.co/\w+'
SPECIAL_CHARS = '&lt;|&lt;|&amp;|#'
def clean(text):
    text = re.sub(HANDLE, ' ', text)
    text = re.sub(LINK, ' ', text)
    text = re.sub(SPECIAL_CHARS, ' ', text)
    return text
Tweets_data['text'] = Tweets_data.text.apply(clean)
Tweets_data.head(10)
# LDA
pip install gensim
from gensim.parsing.preprocessing import preprocess_string
Tweets_list= Tweets_data.text.apply(preprocess_string).tolist()
from gensim import corpora
from gensim.models.ldamodel import LdaModel
dictionary = corpora.Dictionary(Tweets_list)
corpus = [dictionary.doc2bow(text) for text in Tweets_list]
NUM_TOPICS = 5
ldamodel = LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=10)
ldamodel.print_topics(num_words=5)
from gensim.models.coherencemodel import CoherenceModel
def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()
def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        ldamodel = LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=2)
        coherence = calculate_coherence_score(Tweets_list, dictionary, ldamodel)
        yield coherence
min_topics, max_topics = 10,16
coherence_scores = list(get_coherence_values(min_topics, max_topics))
import matplotlib.pyplot as plt
x = [int(i) for i in range(min_topics, max_topics)]
ax = plt.figure(figsize=(10,8))
plt.xticks(x)
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores', fontsize=10);

