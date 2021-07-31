import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

ps = PorterStemmer()

df = pd.read_csv("SMSSpamCollection",sep="\t",names=["label","message"])

Lb = LabelEncoder()
df.label = Lb.fit_transform(df.label)
df = df.drop_duplicates(keep="first")
print(df.shape)
def clean_msg(msg) :
    msg = msg.lower()
    msg = nltk.word_tokenize(msg)
    li1 = []
    for wd in msg :
        if wd.isalnum() :
            li1.append(wd)
    lis2=[]
    for wd in li1 :
        if wd not in stopwords.words("english") and string.punctuation :
            lis2.append(wd)
    li1 = []
    for wd in lis2 :
        li1.append(ps.stem(wd))
    return " ".join(li1)


df["clear_msg"] = df.message.apply(clean_msg)

tfidf = TfidfVectorizer(max_features=3000)
# print("rakesh")

x_tfidf = tfidf.fit_transform(df.clear_msg)
y = df.label.values
x_tfidf_train,x_tfidf_test,y_tfidf_train,y_tfidf_test = train_test_split(x_tfidf,y,test_size=0.2)
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
Mnb = MultinomialNB()
Mnb.fit(x_tfidf_train,y_tfidf_train)
pred_tfidf = Mnb.predict(x_tfidf_test)
print(metrics.confusion_matrix(y_tfidf_test,pred_tfidf))
print(metrics.accuracy_score(y_tfidf_test,pred_tfidf))
print(metrics.precision_score(y_tfidf_test,pred_tfidf))

with open("model.pkl","wb") as f :
    pickle.dump(Mnb,f)

with open("tfidf.pkl","wb") as f :
    pickle.dump(tfidf,f)
