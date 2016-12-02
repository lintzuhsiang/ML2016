# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

#from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from pylab import *
import logging
from optparse import OptionParser
import sys
from time import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text 
my_stop_words = frozenset(["file","an","in","with","and","the","can","you","for","of","from","is","what","on","to","not","how","do","english","using"])
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

location = sys.argv[1]+'title_StackOverflow.txt'
dataset = pd.read_csv(location, sep='\n',encoding='utf-8', header=None,squeeze=True)

# parse commandline arguments
#op = OptionParser()
'''
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()
'''
'''
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
dataset = dataset.apply(lambda x: [porter.stem(w) for w in wordpunct_tokenize(x) if w not in stop_words]).str.join(" ")
'''
# (opts, args) = op.parse_args()
# if len(args) > 0:
#     op.error("this script takes no arguments.")
#     sys.exit(1)
#Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
#categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

# dataset = fetch_20newsgroups(subset='all', categories=categories,
#                              shuffle=True, random_state=42)
'''
print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()
'''
#labels = dataset.target

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
# if opts.use_hashing:
#     if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        #hasher = HashingVectorizer(n_features=opts.n_features,
              #                     stop_words=my_stop_words, non_negative=True,
               #                    norm=None, binary=False)
        #vectorizer = make_pipeline(hasher, TfidfTransformer())
    #else:
        #vectorizer = HashingVectorizer(n_features=opts.n_features,
                #                       stop_words=my_stop_words,
                 #                      non_negative=False, norm='l2',
                  #                     binary=False)
#else:
vectorizer = TfidfVectorizer(max_df=0.4, max_features=10000,
                                 min_df=2, stop_words=my_stop_words,
                                 use_idf=True)
X = vectorizer.fit_transform(dataset)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

#if opts.n_components:
print("Performing dimensionality reduction using LSA")
t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
n_components = 22
svd = TruncatedSVD(n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

print("done in %fs" % (time() - t0))
explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    #print()

true_k = 50


#Do the actual clustering
#if opts.minibatch:
    #km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
    #                    init_size=1000, batch_size=1000, verbose=opts.verbose)
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=5)
                #verbose=opts.verbose)
#else:
#    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
#               verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()

km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()
'''
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
'''
print()


# if not opts.use_hashing:
#     print("Top terms per cluster:")

#     if opts.n_components:
#         original_space_centroids = svd.inverse_transform(km.cluster_centers_)
#         order_centroids = original_space_centroids.argsort()[:, ::-1]
#     else:
#         order_centroids = km.cluster_centers_.argsort()[:, ::-1]

#     terms = vectorizer.get_feature_names()
#     for i in range(true_k):
#         print("Cluster %d:" % i, end='')
#         for ind in order_centroids[i, :10]:
#             print(' %s' % terms[ind], end='')
#         print()


check_data = np.genfromtxt(sys.argv[1]+'check_index.csv', delimiter =',' ,usecols = range(0,3),skip_header = 1)
x_check = check_data[:,1].astype(int)
y_check = check_data[:,2].astype(int)
X = X[:,1:22]
y_predict = km.fit_predict(X,X)
output = (y_predict[x_check]==y_predict[y_check]).astype(int)

out_save=[]
out_save = np.concatenate((np.arange(5000000).reshape(-1,1),output.reshape(-1,1)),axis = 1)

np.savetxt(sys.argv[2],out_save,delimiter = ',' ,fmt ='%i',comments='',header ="ID,Ans")
print('csv saved.')


feat_names = vectorizer.get_feature_names()


#plt.figure(figsize=(12, 12))

for compNum in range(1, 20):
    comp = svd.components_[compNum]
    # Sort the weights in the first component, and get the indeces
    indeces = np.argsort(comp).tolist()
    # Reverse the indeces, so we have the largest weights first.
    indeces.reverse()
    # Grab the top 10 terms which have the highest weight in this component.        
    terms = [feat_names[weightIndex] for weightIndex in indeces[0:10]]    
    weights = [comp[weightIndex] for weightIndex in indeces[0:10]]    
   
    # Display these terms and their weights as a horizontal bar graph.    
    # The horizontal bar graph displays the first item on the bottom; reverse
    # the order of the terms so the biggest one is on top.
    terms.reverse()
    weights.reverse()
    positions = arange(10) + .5    # the bar centers on the y axis
    
    figure(compNum)
    barh(positions, weights, align='center')
    yticks(positions, terms)
    xlabel('Weight')
    title('Strongest terms for component %d' % (compNum))
    grid(True)
    show()
    title('Visualize the K_cluster for component %d' % (compNum))
    plt.scatter(X[:,0],X[:,compNum], c=y_predict, s=50, cmap=plt.cm.Paired)
    plt.colorbar()
    plt.show()
    #plt.subplot(22compNum)
