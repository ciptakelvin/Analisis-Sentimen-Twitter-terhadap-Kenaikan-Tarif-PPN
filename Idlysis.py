# import random
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import precision_score,f1_score,recall_score,accuracy_score
from joblib import dump, load
from datetime import datetime
import numpy as np

#Import Classifier Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from IPython.display import clear_output


class Preprocessor():
    def __init__(self,text,order=""):
        """
        Preprocessing Text: Digunakan untuk membersihkan teks sebelum dilakukan analisis.
        mencakup proses casefolding, filtering
        """
        self.text=text
        self._transform_emoticon()
        self._casefolding()
        self._filtering()
        self._tokenize()
        self._standarize()
        self._stemming()
        self._remove_stop_words()

    def get_text(self):
        return " ".join(self.text)
    
    def _casefolding(self):
        #Mengubah menjadi huruf kecil        
        self.text=self.text.lower()
        
    def _transform_emoticon(self):
        positive_emoji=["😊","🙂","😀","😃","😄","😁","😆","🤣","😂","😊","🥳","🥰","😍","🤩","😘","😚","😙","😋","🤗","🤭",]
        neutral_emoji=["😅","🙃","😉","😇","😎","😗","☺","😛","😜","😝","🤑","🤫","🤔","😐","🤐","🤨","😶","😏","😬","🤥","😌","🤤","😷","🥴","😳","🥺","😮","😯","😲","😦","😧","👻","👽","👾","🤖"]
        negative_emoji=["🤓","🧐","🥲","🤪","😑","😒","🙄","😪","😴","😔","🤒","🤕","🤢","🤮","🤧","🥵","🥶","😵","🤯","😕","😟","🙁","☹","😨","😰","😥","😢","😭","😱","😖","😣","😞","😓","😩","😫","🥱","😤","😡","😠","🤬","😈","👿","💀","☠","💩","🤡","👹","👺",]
        
        for i in positive_emoji:
            self.text=self.text.replace(i," positivemoji ")
        
        for i in neutral_emoji:
            self.text=self.text.replace(i," neutralmoji ")
        
        for i in negative_emoji:
            self.text=self.text.replace(i," negativemoji ")
                
    def _filtering(self):        
        #Url
        self.text=re.sub("https\S+","",self.text)
        self.text=re.sub("http\S+","",self.text)
        self.text=re.sub("\S+\.com\S+","",self.text)
        self.text=re.sub("\S+\.com","",self.text)
        
        #Remove Hashtag
        self.text=re.sub("#\S+","",self.text)
        
        #Remove Mention
        self.text=re.sub("@\S+","",self.text)
        
        #Remove Symbol and Number
        self.text=re.sub("[^A-Za-z\s]"," ",self.text)
        
        #Remove Spacing
        self.text=re.sub("\s+"," ",self.text)
        self.text=re.sub("^\s","",self.text)
        self.text=self.text

        
    def _tokenize(self):
        #Membagi kata
        self.text=self.text.split(" ")
    
   
    def _standarize(self):        
        #Mengubah menjadi kata baku
        j={}
        with open("standard_word.csv","r") as file:
            data=csv.reader(file,delimiter=",")
            for k,i in enumerate(data):
                if k==0: continue
                j[i[0]]=i[1]
                
        for k,t in enumerate(self.text):
            if t in j:
                self.text[k]=j[t]
        
    
    def _stemming(self):
        #Mengubah menjadi kata dasar
        factory=StemmerFactory()
        stemmer=factory.create_stemmer()
        
        self.text=stemmer.stem(" ".join(self.text))
        self._tokenize()

    def _remove_stop_words(self):
        stop_words = set(stopwords.words('indonesian'))
        add_stopwords=[]
        for i in add_stopwords:
            stop_words.append(i)
        w=[]
        for i in self.text:
            if i in stop_words:
                w.append("")
            else:
                w.append(i)
                
        self.text=w

        
    
class Analyzer():  
    def __init__(self):
        """
        Membuat model dan melakukan prediksi
        """
        self.t_size=0.2 #test size
    
    def predict(self,training_data:pd.DataFrame,data_to_predict:pd.DataFrame):
        #Train and Predict Directly
        model=self.create_model(training_data)
        return self.predict_by_model(model,data_to_predict)
    
    def predict_by_model(self,model,data:pd.DataFrame):

        #Output Data
        target_column:int=len(data.columns)-1
        X=data.iloc[:,data.columns!=data.columns[target_column]]
        y=data[data.columns[target_column]]
        prediction=model.predict(X)
        return prediction
    
        
    def create_model(self,data:pd.DataFrame,is_save:bool=False):
        target_column:int=len(data.columns)-1
        X=data.iloc[:,data.columns!=data.columns[target_column]]
        y=data[data.columns[target_column]]
        
        models_used=[
            KNeighborsClassifier(),
            SVC(),
            GaussianNB(),
            MultinomialNB(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            AdaBoostClassifier()
        ]
        
        
        max_accuracy=0
        for i in models_used:
            accuracies=[]
            for j in range(10):
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=self.t_size)
                i.fit(X_train,y_train)
                prediction=i.predict(X_test)
                accuracies.append(accuracy_score(prediction,y_test))
                print("Testing.. Accuracy Score: ",accuracy_score(y_test,prediction))
            
            accuracy=np.average(accuracies)
            if accuracy>max_accuracy:
                max_accuracy=accuracy
                model=i
            print(i,prediction)
            print("Average Accuracy Score:",accuracy,"\n")

        print("Training Data Size: ",data.shape[0])
        print("Training Test Size: ",self.t_size*100,"%")
        print("Model used: "+model.__class__.__name__+" model. Accuracy: "+str(max_accuracy))
        if is_save:
            dump(model,"models/"+model.__class__.__name__+" "+str(datetime.now()).replace(":","")+".joblib")
        return model
    
