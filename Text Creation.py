# This work was done for Analytics Edge, a class at MIT applying Machine Learning to real-world applications.
# This was completed as a portion of our final semester project with Lucia Perez-Sanchez


import pandas as pd
import numpy as np
import re
from string import punctuation, digits, printable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import KMeans

loans = pd.read_csv("C:/Users/subha/Dropbox (MIT)/Analytics Edge Project/new_join_clean.csv")

print(loans.shape)

loans = loans[loans['use'].notnull()]
# remove punctuation
loans["clean_txt"] = [word.replace(',',' ') for word in loans.use]
loans["clean_txt"] = [word.replace('.',' ') for word in loans.clean_txt]

loans['usage_length']=loans.use.str.len()
loans["females"] = [word.replace(',',' ').split().count("female") for word in loans.borrower_genders]
loans["males"] = [word.replace(',',' ').split().count("male") for word in loans.borrower_genders]

vectoriser = TfidfVectorizer(stop_words = 'english')
loans['loansVect'] = list(vectoriser.fit_transform(loans['clean_txt']).toarray())

dtm = list(vectoriser.fit_transform(loans['clean_txt']).toarray())
pd.DataFrame(vectoriser.get_feature_names()).to_csv("DTM_columns.csv")

dtm_df= pd.DataFrame(dtm)
dtm_df.to_csv("C:/Users/subha/Dropbox (MIT)/Analytics Edge Project/dtm.csv")

DTM_loans = pd.read_csv("C:/Users/subha/Dropbox (MIT)/Analytics Edge Project/dtm.csv")

DTM_colnames = pd.read_csv("/Users/analucia/Dropbox (MIT)/Analytics Edge Project/DTM_columns.csv")

DTM_loans = DTM_loans.drop(["Unnamed: 0"],1)

scaler = MinMaxScaler()
scaler.fit(DTM_loans)
df_c = scaler.transform(DTM_loans)
get_ipython().run_line_magic('matplotlib', 'inline')

loans['loan_funded_indicator'] = loans['funded_amount'] - loans['loan_amount']
loans.loc[loans.loan_funded_indicator >= 0, 'loan_funded_indicator'] = 1.0
loans.loc[loans.loan_funded_indicator < 0, 'loan_funded_indicator'] = 0.0
loans['use']= loans.use.str.lower()
loans['usage_length']=loans.use.str.len()
loans["females"] = [word.replace(',',' ').split().count("female") for word in loans.borrower_genders]
loans["males"] = [word.replace(',',' ').split().count("male") for word in loans.borrower_genders]

k = 4
kmeans = KMeans(n_clusters=k, random_state=0).fit(df_c)
labels_km = kmeans.labels_
np.unique(labels_km, return_counts=True)
cluster_0 = loans[labels_km == 0]
cluster_1 = loans[labels_km == 1]
cluster_2 = loans[labels_km == 2]
cluster_3 = loans[labels_km == 3]
# see how many fall into each bucket
np.unique(labels_km, return_counts=True)

cluster_assignments_dict = {}
clean_dictionary = dict(zip(loans['id'], loans['loan_amount']))
for i in set(labels_km):
    #print i
    current_cluster = [list(clean_dictionary)[x] for x in np.where(labels_km == i)[0]]
    cluster_assignments_dict[i] = current_cluster
plt.hist(labels_km, bins=10)
plt.show()

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_words(words_list, stemmer):
    return [stemmer.stem(word) for word in words_list]

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_words(tokens, stemmer)
    return stems
dict_cluster = {}
dict_cluster[0] = cluster_0['clean_txt']
dict_cluster[1] = cluster_1['clean_txt']
dict_cluster[2] = cluster_2['clean_txt']
dict_cluster[3] = cluster_3['clean_txt']

cluster_themes_dict = {}

for i in range(k):
    current_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    current_tfs = current_tfidf.fit_transform(dict_cluster[i])

    current_tf_idfs = dict(zip(current_tfidf.get_feature_names(), current_tfidf.idf_))
    tf_idfs_tuples = current_tf_idfs.items()
    cluster_themes_dict[i] = sorted(tf_idfs_tuples, key=lambda x: x[1])[:15]
cluster_themes_dict

## 0:  goods
## 1: health (water, medicine, safety)
## 2: education
## 3: agriculture

from sklearn.decomposition import PCA

plt.figure(figsize = (11, 9))

target_names = ['1. general agribusiness', '2. health and water', '3. education','4. crop-specific agricultural']
colors = ['lightgreen', 'cyan', 'mediumorchid', 'lightcoral']
lw = 4

for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
    plt.scatter(X_pca[labels_km == i, 0], X_pca[labels_km == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1, fontsize = 18)
plt.title('PCA: clustered loans', fontsize =22)