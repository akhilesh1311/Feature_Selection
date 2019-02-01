from __future__ import division
import shutil
import os
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import time
from collections import Counter
import math
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")
from stemming.porter2 import stem
from nltk.corpus import stopwords
import itertools
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
import string
import nltk
from collections import Counter
from scipy.stats import norm
import heapq
from scipy import sparse
from collections import Counter
from scipy.stats import norm
import cPickle as pickle
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2
from termcolor import colored

cwd = os.getcwd()
'''
Place the Classic dataset (data files) on creating a folder by name "classic".
Create another folder "Classic_Dataset" and 4 sub folders by name "med", "cran", "cicsi", "cacm".
 
The hardcoded number(a, b, c, d) of files will be copied from the "classic" to corresponding folders ("cacm", "med", "cisi", "cran").

'''
dirPath = cwd + "/reuters_lite/acq"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

dirPath = cwd + "/reuters_lite/crude"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

dirPath = cwd + "/reuters_lite/earn"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

dirPath = cwd + "/reuters_lite/grain"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

dirPath = cwd + "/reuters_lite/interest"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

dirPath = cwd + "/reuters_lite/money-fx"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

dirPath = cwd + "/reuters_lite/ship"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

dirPath = cwd + "/reuters_lite/trade"
fileList = os.listdir(dirPath)
for fileName in fileList:
	os.remove(dirPath+"/"+fileName)

#CACM, CISI, CRAN, MED (total number of docs in corresponding classed : 3204, 1460, 1400, 1033)
a = 400
#a = 2291
b = 65
#b = 373
c = 200
#c = 3922
d = 45
#d = 50
e = 65
#e = 270
f = 60
#f = 292
g = 55
#g = 143
h = 50
#h = 325
#Hard code the "M" percentage value i.e.: the top 'M%' features you wish to select from each class.
top_num = 90
manual_nfr = 0.6
print colored(top_num,color='cyan')
print colored(manual_nfr,color='cyan')
auto_nfr = np.std([a,b,c,d])
print auto_nfr

for j in range(0, a):
	dest = cwd + "/reuters_lite/acq"
	name = "/reuters/Reuters_Dataset/acq/acq_" + (str)(j) + ".txt"
	source = cwd + name
	shutil.copy2(source, dest)

for j in range(0, b):
	dest = cwd + "/reuters_lite/crude"
	name = "/reuters/Reuters_Dataset/crude/crude_" + (str)(j) + ".txt"
	source = cwd + name
	shutil.copy2(source, dest)

for j in range(0, c):
	dest = cwd + "/reuters_lite/earn"
	name = "/reuters/Reuters_Dataset/earn/earn_" + (str)(j) + ".txt"
	source = cwd + name
	shutil.copy2(source, dest)

for j in range(0, d):
	dest = cwd + "/reuters_lite/grain"
	name = "/reuters/Reuters_Dataset/grain/grain_" + (str)(j) + ".txt"
	source = cwd + name
	shutil.copy2(source, dest)

for j in range(0, e):
	dest = cwd + "/reuters_lite/interest"
	name = "/reuters/Reuters_Dataset/interest/interest_" + (str)(j) + ".txt"
	source = cwd + name
	shutil.copy2(source, dest)

for j in range(0, f):
	dest = cwd + "/reuters_lite/money-fx"
	name = "/reuters/Reuters_Dataset/money-fx/money-fx_" + (str)(j) + ".txt"
	source = cwd + name
	shutil.copy2(source, dest)

for j in range(0, g):
	dest = cwd + "/reuters_lite/ship"
	name = "/reuters/Reuters_Dataset/ship/ship_" + (str)(j) + ".txt"
	source = cwd + name
	shutil.copy2(source, dest)

for j in range(0, h):
	dest = cwd + "/reuters_lite/trade"
	name = "/reuters/Reuters_Dataset/trade/trade_" + (str)(j) + ".txt"
	source = cwd + name
	shutil.copy2(source, dest)

#Load The files/dataset
load_path = cwd + "/reuters_lite"
#load_path_baklol = cwd + "/sample_collection"
dataset=load_files(load_path, description=None, categories=None, load_content=True, shuffle=False, encoding=None, decode_error='strict', random_state=0)
#dataset=load_files(load_path_baklol, description=None, categories=None, load_content=True, shuffle=False, encoding=None, decode_error='strict', random_state=0)

#Class names and assigned numbers
class_names= list(dataset.target_names) #converting to list redundant
class_num = len(class_names)
class_numbers = []
for i in range(0, len(class_names)):
	class_numbers.append(i)

#Document class labels
d_labels = dataset.target
#Data from the dataset
vdoc = dataset.data
#print vdoc
#print d_labels
#print dataset.target_names

#Stemming the words
stemmer = PorterStemmer() #PorterStemmer class redundant
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stem(item))
    return stemmed

#Tokenizing each word
def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

#Count Tokenizer --> Finding doc-term frequency matrix
vec = CountVectorizer(tokenizer=tokenize, stop_words='english')	#important class
data = vec.fit_transform(vdoc).toarray()		#transforms the dataset to dense matrix of document*term
sparse_data = csr_matrix(data)
voc = vec.get_feature_names()		#get the names of terms
data_transpose = data.transpose()
#print data_transpose

#Finding Vocabulary
vocabulary = voc
voc_num = len(vocabulary)
doc_num = data.shape[0]		#shape[0] = number of rows in data; shape[1] = number of columns in data

#final doc-term tfidf vector
tf_vec = TfidfTransformer(use_idf=True).fit(data)
vectors = tf_vec.transform(data)		# vector is (doc-id,term-id) to idf-value mapping

class_doc = []		#print this class_doc variable
class_doc.append([i for i, j in enumerate(d_labels) if j == 0])		#class_doc is class to document mapping, ie, class1->doc1,doc2,etc
class_doc.append([i for i, j in enumerate(d_labels) if j == 1])
class_doc.append([i for i, j in enumerate(d_labels) if j == 2])
class_doc.append([i for i, j in enumerate(d_labels) if j == 3])
class_doc.append([i for i, j in enumerate(d_labels) if j == 4])
class_doc.append([i for i, j in enumerate(d_labels) if j == 5])
class_doc.append([i for i, j in enumerate(d_labels) if j == 6])
class_doc.append([i for i, j in enumerate(d_labels) if j == 7])
#class_doc.pop()

cl_voc = []
for item in data:		#construction of forward matrix, ie, document1->termid1,termid2,etc
	cl_voc.append([i for i, j in enumerate(item) if j != 0])
#print cl_voc
term_to_doc = data.transpose()

term_to_class = []		#construction of term to class mapping, ie, term1->class1,class1,class2,etc
for i in range(0,term_to_doc.shape[0]):
	s = []
	for j in range(0,term_to_doc.shape[1]):
		if(term_to_doc[i][j] != 0):
			for m in range(0,len(class_doc)):
				s1 = [m for x in class_doc[m] if x == j]
				if(len(s1) != 0):
					s.extend(s1)
	term_to_class.append(s)
term_to_class_np = np.array(term_to_class)
#print term_to_class_np

class_voc = []		#construction of class to term mapping(class1-> term1,term2,etc) 
for item in class_doc:			#from class_doc(class to document mapping) and cl_voc(document to term mapping)
	temp = [cl_voc[x] for x in item]
	class_voc.append(list(set(list(itertools.chain.from_iterable(temp)))))
#class_voc.pop()
dense_matrix_class_term = []
for item in class_voc:
	temp = []
	for i in range(0,len(voc)):
		if i in item:
			temp.extend([1])
		else:
			temp.extend([0])
	dense_matrix_class_term.append(temp)
#print class_voc
sparse_matrix_class_term = csr_matrix(dense_matrix_class_term)
#print sparse_matrix_class_term

class_vec = []		#construction of document to term-idf mapping classwise ([[class1],[class2],[class3],[class4]]), 
for i in range(0,len(class_doc)):		#where class1 = [[doc1],[doc2],[doc3],etc] where doc1 = [term-idf1,term-idf2,etc]
	s = []										#and then storing it in CSR matrix format
	for item in class_doc[i]:
		s.append(vectors.getrow(item).toarray().tolist()[0])
	class_vec.append(csr_matrix(s))
#print class_vec


def gini_Index(class_to_docs,docs_to_term,term_to_class):
	gini_terms = []
	for i in range(0,len(term_to_class)):
		sum = 0.0
#		temp1 = np.array(term_to_class)[i].tolist()
		temp1 = term_to_class[i]
		for j in range(0,len(class_to_docs)):
			temp2 = class_to_docs[j]
			sum = sum + math.pow((len([x for x in temp2 if i in docs_to_term[x]]))/float((len(temp2))),2) * math.pow((len([x for x in temp1 if x == j]))/float((len(temp1))),2)
		gini_terms.append(sum)
	return gini_terms
	
def poisson_distribution(data_transpose,docs_to_term,class_to_docs,test_size):
	doc_no = len(docs_to_term)
	class_no = len(class_to_docs)
	n = math.floor(doc_no*(1 - test_size))
	poisson_terms = []
	for i in range(0,voc_num):
		s = 0
		for k in range(0,doc_no):
			s = s + data_transpose[i][k]
		f = s
		lambd = float(f)/float(n)
		dpt = 0
		for j in range(0,class_no):
			a = len([x for x in class_to_docs[j] if i in docs_to_term[x]])
			b = len(class_to_docs[j]) - a
			sum = 0
			for k in range(0,class_no):
				if(k != j):
					sum = sum + len([x for x in class_to_docs[k] if i in docs_to_term[x]])
			c = sum
			d = doc_no - c - a
			nC = len(class_to_docs[j])
			nCbar = doc_no - nC
			abar = nC*(1-math.exp(-lambd))
			bbar = nC*math.exp(-lambd)
			cbar = nCbar*(1-math.exp(-lambd))
			dbar = nCbar*math.exp(-lambd)
			term1 = math.pow(a-abar,2)/abar
			term2 = math.pow(b-bbar,2)/bbar
			term3 = math.pow(c-cbar,2)/cbar
			term4 = math.pow(d-dbar,2)/dbar
			pC = len(class_to_docs[j])/doc_no
			dpt = dpt + pC*(term1+term2+term3+term4)
		poisson_terms.append(dpt)
	return poisson_terms
			
#Odds Ratio
def odds_ratio(class_to_docs,docs_to_term):
	odds_ratio_term_class = []
	total_length = len(docs_to_term)
	for i in range(0,voc_num):
		s = []
		for j in range(0,len(class_to_docs)):
			PtCj = float(len([x for x in class_to_docs[j] if i in docs_to_term[x]]))/float(len(class_to_docs[j]))
			remaining_length = total_length - len(class_to_docs[j])
			remaining_length_term = 0
			for k in range(0,len(class_to_docs)):
				if(k != j):
					remaining_length_term = remaining_length_term + len([x for x in class_to_docs[k] if i in docs_to_term[x]])
			PtCbarj = float(remaining_length_term)/float(remaining_length)
			s1 = math.log((PtCj*(1-PtCbarj) + 0.01)/((1-PtCj)*PtCbarj + 0.01))	#add 0.01 to both num and denom to eliminate division by 0 errors
			s.append(s1)
		odds_ratio_term_class.append(s)
	return odds_ratio_term_class
	
#Odds Ratio Label - 0 is negative and 1 is positive
def odds_ratio_label(odds_ratio_term_class):
	odds_ratio_final_label = []
	odds_ratio_pos_neg = []
	diff = 0.0
	for i in range(0,len(odds_ratio_term_class)):
		if(max(odds_ratio_term_class[i]) < abs(min(odds_ratio_term_class[i]))):
			odds_ratio_final_label.append(odds_ratio_term_class[i].index(min(odds_ratio_term_class[i])))
			odds_ratio_pos_neg.append(0)
		else:
			odds_ratio_final_label.append(odds_ratio_term_class[i].index(max(odds_ratio_term_class[i])))
			odds_ratio_pos_neg.append(1)
	return odds_ratio_final_label, odds_ratio_pos_neg
	
start1 = time.time()
ans1_gini = gini_Index(class_doc,cl_voc,term_to_class)
ans1_poisson = poisson_distribution(data_transpose,cl_voc,class_doc,0.4)
print "I step time %f" % (time.time() - start1)

ans2 = odds_ratio(class_doc,cl_voc)

ans3,ans4 = odds_ratio_label(ans2)

#IGFSS
def igfss(gini_index_term,fs,odds_ratio_label,nfrs,no_of_classes,odds_ratio_pos_neg):
	ffs = []
	sl =  heapq.nlargest(int(len(gini_index_term)), zip(gini_index_term, itertools.count()))
	fs_each_class = fs/no_of_classes
	neg_each_class = nfrs * fs_each_class
	pos_each_class = (1-nfrs) * fs_each_class
	for i in range(0,no_of_classes):
		temp = []
		j = 0
		while(len(temp) < pos_each_class and j < len(gini_index_term)):
			if(sl[j][1] > 0 and i == odds_ratio_label[sl[j][1]] and odds_ratio_pos_neg[sl[j][1]] == 1):
				temp.extend([sl[j][1]])
				sl[j] = list(sl[j])
				sl[j][1] = -sl[j][1]
				sl[j] = tuple(sl[j])
			j = j+1
		j = 0
		while(len(temp) < fs_each_class and j < len(gini_index_term)):
			if(sl[j][1] > 0 and i == odds_ratio_label[sl[j][1]] and odds_ratio_pos_neg[sl[j][1]] == 0):
				temp.extend([sl[j][1]])
				sl[j] = list(sl[j])
				sl[j][1] = -sl[j][1]
				sl[j] = tuple(sl[j])
			j = j+1
		ffs.extend(temp)
	
	diff = fs - len(ffs)
	i = 0
	while(diff > 0 and i < len(gini_index_term)):
		if(sl[i][1] > 0):
			ffs.extend([sl[i][1]])
			diff = diff-1
		i = i+1
	return ffs

def chi_square(vectors,d_labels,top_num):
	ans5,pval = chi2(vectors,d_labels)
	sl =  heapq.nlargest(int(top_num*0.01*vectors.shape[1]), zip(ans5, itertools.count()))
	ans = []
	for item in sl:
		ans.append(item[1])
	return ans

ans5_gini = igfss(ans1_gini,voc_num*0.01*top_num,ans3,manual_nfr,4,ans4)
ans5_poisson = igfss(ans1_poisson,voc_num*0.01*top_num,ans3,manual_nfr,4,ans4)
print "Feature Selection time taken %f \n" % (time.time() - start1)
ans6 = chi_square(vectors,d_labels,top_num)

vectors = vectors.transpose()
gini_s = []
for item in ans5_gini:
	gini_s.append(vectors.getrow(item).toarray().tolist()[0])
#print s	#s is [[term1],[term2],etc] where each [term1] = [doc1,doc2,doc3,etc] where doc1,doc2 denote the term_tfidf value of term1 in document1,in 
			#document2 and so on

chi_s = []
for item in ans6:
	chi_s.append(vectors.getrow(item).toarray().tolist()[0])

chi_vectors = csr_matrix(chi_s)
chi_vectors = chi_vectors.transpose()

poisson_s = []
for item in ans5_poisson:
	poisson_s.append(vectors.getrow(item).toarray().tolist()[0])

poisson_vectors = csr_matrix(poisson_s)
poisson_vectors = poisson_vectors.transpose()

gini_vectors = csr_matrix(gini_s)
#print new_vector
gini_vectors = gini_vectors.transpose()
#print new_vector

vectors = vectors.transpose()

def Classification(X_train, X_test, y_train, y_test):
	print "\n----------SVM linear Kernel--------\n"
	start1 = time.time()
	clf = svm.SVC(kernel='linear', C=1)
	clf.fit(X_train, y_train)
	print "Training time taken %f \n" % (time.time() - start1)

	start2 = time.time()
	y_pred = clf.predict(X_test).tolist()
	print "Testing time taken %f \n" % (time.time() - start2)
	print "Classification report:\n"
	print metrics.classification_report(y_test, y_pred)
	print "Accuracy:\n"
	print metrics.accuracy_score(y_test, y_pred)
	
	print "\n----------KNN--------\n"
	start1 = time.time()
	clf2 = KNeighborsClassifier(n_neighbors = 2,algorithm = 'auto')
	clf2.fit(X_train, y_train)
	print "Training time taken %f \n" % (time.time() - start1)

	start2 = time.time()
	y_pred = clf2.predict(X_test).tolist()
	print "Testing time taken %f \n" % (time.time() - start2)
	print "Classification report:\n"
	print metrics.classification_report(y_test, y_pred)
	print "Accuracy:\n"
	print metrics.accuracy_score(y_test, y_pred)
	
	print "\n----------Naive_Bayes--------\n"
	start1 = time.time()
	clf3 = MultinomialNB()
	clf3.fit(X_train, y_train)
	print "Training time taken %f \n" % (time.time() - start1)

	start2 = time.time()
	y_pred = clf3.predict(X_test).tolist()
	print "Testing time taken %f \n" % (time.time() - start2)
	print "Classification report:\n"
	print metrics.classification_report(y_test, y_pred)
	print "Accuracy:\n"
	print metrics.accuracy_score(y_test, y_pred)

print colored('\nNo feature Selection',color='yellow',on_color=None,attrs=['bold','underline'])
print "\nClassifying Original Vectors Without feature selection"
print "\nVectors Shape(Doc X Terms): %.1f x %.1f" % (vectors.shape[0], vectors.shape[1])
X1 = vectors.toarray()
y1 = d_labels
X_train1, X_test1, y_train1, y_test1 = cross_validation.train_test_split(X1, y1, test_size=0.4, random_state=0)
Classification(X_train1, X_test1, y_train1, y_test1)

#These X2 is the doc-term matrix and y2 is corresponding class labels (target values)
print colored('\nChi2 feature Selection',color='yellow',on_color=None,attrs=['bold','underline'])
print "\nClassifying Feature Reduced Vectors with Chi2"
print "\nNew Vectors Shape(Doc X Terms): %.1f x %.1f" % (chi_vectors.shape[0], chi_vectors.shape[1])
X2 = chi_vectors.toarray()
y2 = d_labels
X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X2, y2, test_size=0.4, random_state=0)
Classification(X_train2, X_test2, y_train2, y_test2)

print colored('\nOriginal IGFSS feature Selection',color='yellow',on_color=None,attrs=['bold','underline'])
print "\nClassifying Feature Reduced Vectors with Gini in IGFSS"
print "\nNew Vectors Shape(Doc X Terms): %.1f x %.1f" % (gini_vectors.shape[0], gini_vectors.shape[1])
X2 = gini_vectors.toarray()
y2 = d_labels
X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X2, y2, test_size=0.4, random_state=0)
Classification(X_train2, X_test2, y_train2, y_test2)

print colored('\nModified IGFSS feature Selection',color='yellow',on_color=None,attrs=['bold','underline'])
print "\nClassifying Feature Reduced Vectors with Modified IGFSS using Poisson"
print "\nNew Vectors Shape(Doc X Terms): %.1f x %.1f" % (poisson_vectors.shape[0], poisson_vectors.shape[1])
X2 = poisson_vectors.toarray()
y2 = d_labels
X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(X2, y2, test_size=0.4, random_state=0)
Classification(X_train2, X_test2, y_train2, y_test2)


