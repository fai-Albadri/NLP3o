#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module

Contains the class TextAnalyser to handle a text

NLP3O is a word game between NLP (Natural Language Processing) and C3PO

Created on Tue Apr 25 15:10:58 2017

@author: Mashimo
"""
from .inputhandler import readStopwords
import nltk
import pandas as pd
from camel_tools.dialectid import DialectIdentifier
from camel_tools.sentiment import SentimentAnalyzer
from camel_tools.utils import normalize

#2:36
# init the model
#camel

from nltk.stem import WordNetLemmatizer
#add camel tools
class TextAnalyser:
        # stemming values
    NO_STEMMING = 0
    STEM = 1
    LEMMA = 2

    'Text object, with analysis methods' 
    def __init__(self, inputText, language = "EN"):
        self.text = inputText
        self.tokens = []
        self.sentences = []
        self.language = language
        self.stopWords = set(readStopwords(language))
        #####
    def getMostCommonCities(self, n=5):
        """ get the n most common words in the text;
        n is the optional paramenter"""
        from collections import Counter
        did = DialectIdentifier.pretrained()
        predictions = did.predict([self.text],output='city')
        sorted_cities = sorted(predictions[0].scores.items(),key=lambda x: x[1],reverse=True)
        sorted_cities = [(city,round(score*100,3)) for city, score in sorted_cities]
        return sorted_cities[:n]
    def getMostCommonRegion(self, n=5):
        """ get the n most common words in the text;
        n is the optional paramenter"""
        from collections import Counter
        did = DialectIdentifier.pretrained()
        predictions = did.predict([self.text],output='region')
        sorted_region = sorted(predictions[0].scores.items(),key=lambda x: x[1],reverse=True)
        sorted_region = [(region,round(score*100,3)) for region, score in sorted_region]
        return sorted_region[:n]

    def getMostCommonCountry(self, n=5):
        """ get the n most common words in the text;
        n is the optional paramenter"""
        from collections import Counter
        did = DialectIdentifier.pretrained()
        predictions = did.predict([self.text],output='country')
        sorted_country = sorted(predictions[0].scores.items(),key=lambda x: x[1],reverse=True)
        sorted_country = [(country,round(score*100,3)) for country, score in sorted_country]
        return sorted_country[:n]
        #####

    def getPositiveORNegative(self):
        sa = SentimentAnalyzer.pretrained()

# Predict the sentiment of a single sentence
        sentiment = sa.predict_sentence(self.text)
        #sentiments = sa.predict(sentiment)
        return sentiment

    def getPositiveORNegative2(self):
        sa = SentimentAnalyzer.pretrained()
# Predict the sentiment of a single sentence
        #sentiment = sa.predict_sentence(self.text)
        sentences = self.text.split('\n')
        sentiments = sa.predict(sentences)
        return sentiments

    def getNormalization(self):
        a=normalize.normalize_alef_ar(self.text)
        b=normalize.normalize_teh_marbuta_ar(a)
        
        return b

    def length(self):
        """ return length of text in chars """
        return len(self.text)
        
    def tokenise(self):
        """ split the text into tokens, store and returns them """
        self.tokens = self.text.split() # split by space; return a list
    
    def tokeniseNLTK(self):
        self.tokens = nltk.word_tokenize(self.text)
    
    def getTokens(self):
        """ returns the tokens (need to be previously tokenised) """
        return len(self.tokens)
    
    def splitSentences(self):
        self.sentences = nltk.sent_tokenize(self.text)
    
    def getSentences(self):
        """ returns the sentences (need to be previously split) """
        return len(self.sentences)
    
    def removePunctuation(self):
        """ remove punctuation from text"""
        import re

        self.text = re.sub(r'([^\s\w_]|_)+', '', self.text.strip()) # remove punctuation

    def removeStopWords(self):
        """ remove stop words from text.
        Stopwords are defined at initialisation based on language.
        Only one set of stopwords is possible (no language mix)"""
        self.tokens = [token for token in self.tokens if token not in self.stopWords]
    

    def lemmatiseVerbs(self):

        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(w,'v') for w in self.tokens]

        #return len(set(lemmatized))
    
    def stemTokens(self):
        porter = nltk.PorterStemmer()
        return [porter.stem(t) for t in self.tokens]

    def preprocessText(self, lowercase=True, removeStopWords=False, stemming=NO_STEMMING):
        """ pre-process the text:
            1. lower case 
            2. remove punctuation
            3. tokenise the text
            4. remove stop words"""
            
        self.splitSentences()
        
        if lowercase:
            self.text = self.text.lower()
    
        self.removePunctuation()

        self.tokenise()  
        
        if removeStopWords:
            self.removeStopWords()
        
            # if stemming then do it
        if stemming == TextAnalyser.STEM:
            self.tokens = self.stemTokens()
        elif stemming == TextAnalyser.LEMMA:
            self.tokens = self.lemmatiseVerbs()


        
    def uniqueTokens(self):
        """ returns the unique tokens"""
        return (len(set(self.tokens)))
    
    def getMostCommonWords(self, n=10):
        """ get the n most common words in the text;
        n is the optional paramenter"""
        from collections import Counter

        wordsCount = Counter(self.tokens) # count the occurrences
    
        return wordsCount.most_common()[:n]
    
    def getMostCommonWordsNLTK(self, n=10):
        """ get the n most common words in the text;
        n is the optional paramenter"""
        # Calculate frequency distribution
        fdist = nltk.FreqDist(self.tokens)

        # Output top 20 words

        return fdist.most_common(n)

    def findLongest(self):
      #Find the longest word in text1 and that word's length.
        longest = max(self.tokens, key=len)
        return (longest, len(longest))


    def findSentences(self):
    
        sentences = nltk.sent_tokenize(self.text)
        return len(self.tokens) / len(sentences)
    
    
    
