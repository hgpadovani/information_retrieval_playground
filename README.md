# Search Engine Based on BM25

- This repository implements the BM25 algorithm in a Search Engine.
- The algorithm is evaluated in the CISI dataset with trec_eval-like metrics.

# POC examples
## Run the entire System

```python
# Importing modules
import DataLoader
import PreProcessing
import SearchEngine
import BM25
import nltk
from rank_bm25 import BM25Okapi
from evaluate import load

# DataLoader
PATH = 'path_to_data' 
dataloader = DataLoader(PATH)

# PreProcessing
stemmer = nltk.stem.PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')
preprocessing = PreProcessing(stopwords = stopwords, stemmer = stemmer)

# Algorithm
algo1 = BM25Okapi
algo2 = BM25()

# trec eval
trec_eval = load("trec_eval")

# SearchEngine1
search_engine_okapi = SearchEngine(dataloader, preprocessing, algo1, trec_eval)
most_relevant_docs_okapi, df_metrics_okapi = search_engine_okapi.run()

# SearchEngine2
search_engine_scratch = SearchEngine(dataloader, preprocessing, algo2, trec_eval)
most_relevant_docs_scratch, df_metrics_scratch = search_engine_scratch.run()
```

## Run a specific query
```python
# Defining query
query = '''
What problems and concerns are there in making up descriptive titles? 
What difficulties are involved in automatically retrieving articles from approximate titles? 
What is the usual relevance of the content of articles to their titles?'''

# Retrieving documents for both algorithm implementations 
retrieved_docs = search_engine.retrieve_docs_from_query(query, n=10)
````
