import re
class Analyzer():
    def __init__(self,text:str):
        self.text=text
    
    def pre(self)->str:
        """
        Preprocessing Text: Digunakan untuk membersihkan teks sebelum dilakukan analisis.
        mencakup proses casefolding, filtering
        """
        self.casefolding()
        self.filtering()
        return self.text

    def casefolding(self):
        self.text=self.text.lower()
    
    def filtering(self):
        #Url
        self.text=re.sub("https\S+","",self.text)
        self.text=re.sub("http\S+","",self.text)
        self.text=re.sub("\S+\.com\S+","",self.text)
        self.text=re.sub("\S+\.com","",self.text)
        
        #Hashtag
        self.text=re.sub("#\S+","",self.text)
        
        #Mention
        self.text=re.sub("@\S+","",self.text)
        
        #Symbol and Number
        self.text=re.sub("[^A-Za-z\s]","",self.text)
        
        #Spacing
        self.text=re.sub("\s+"," ",self.text)
        self.text=re.sub("^\s","",self.text)
        self.text=self.text
