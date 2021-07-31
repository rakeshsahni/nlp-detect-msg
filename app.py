import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from flask import Flask,render_template,request,redirect
ps = PorterStemmer()


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

with open("tfidf.pkl","rb") as f1 :
    tfidf = pickle.load(f1)

with open("model.pkl","rb") as f :
    model = pickle.load(f)

app = Flask(__name__)


@app.route("/",methods=["POST","GET"])
def Home() :
    if(request.method == "POST") :
        txt = request.form["message"]
        # print(txt)
        final_str = clean_msg(txt) 
        final_y = tfidf.transform([final_str])
            # print(final_y.shape)
        
            # 0 - ham 1 - spam
        pred = model.predict(final_y)[0]
        # print(pred)
        hamorspam = ""
        if(pred == 1) :
            hamorspam="☝️ Spam"
        else :
            hamorspam="☝️ Ham" 
                
        return render_template("index.html",txt=txt,hamorspam=hamorspam)
    else :
        return render_template("index.html",txt="",hamorspam="")


@app.route('/<path:path>')
def catch_all(path):
    return redirect("/")

if __name__ == "__main__" :
    app.run(debug=True)