#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routing module for NLP3O app

Created on Tue Apr 18 17:48:15 2017

@author: Mashimo
"""

from app import app
from flask import render_template, flash, request
from .forms import InputTextForm
from .nlp3o import TextAnalyser
from .inputhandler import getSampleText

    # Submit button in web for pressed
@app.route('/', methods=['POST'])
@app.route('/index', methods=['POST'])
def manageRequest():

      # some useful initialisation
    theInputForm = InputTextForm()
    userText = "and not leave this empty!"
    typeText = "You should write something ..."
    language = "EN"

      # POST - retrieve all user submitted data

    inputFromBook = request.form['book'] # which text?

    # DEBUG flash('the book selected is: %s' % inputFromBook)

    if True:
  #  theInputForm.validate_on_submit():
        userText = theInputForm.inputText.data
        typeText = "Your own text"
        language = request.form['lang'] # which language?

    # DEBUG flash('read:  %s' % typeText)
    
    stemmingEnabled = request.form.get("stemming")
    stemmingType = TextAnalyser.NO_STEMMING
    
    if stemmingEnabled:
        if request.form.get("engine"):
            stemmingType = TextAnalyser.STEM
        else:
            stemmingType = TextAnalyser.LEMMA
    else:
        stemmingType = TextAnalyser.NO_STEMMING


    #flash('read:  %s' % stemmingEnabled)
# start analysing the text
        myText = TextAnalyser(userText, language) # new object

        myText.preprocessText(lowercase = theInputForm.ignoreCase.data,
                              removeStopWords = theInputForm.ignoreStopWords.data,
                              stemming = stemmingType)
        # display all user text if short otherwise the first fragment of it
        if len(userText) > 99:
            fragment = userText[:99] + " ..."
        else:
            fragment = userText

              # check that there is at least one unique token to avoid division by 0
        if myText.uniqueTokens() == 0:
            uniqueTokensText = 1
        else:
            uniqueTokensText = myText.uniqueTokens()
          # Which kind of user action ?
    if 'TA'  in request.form.values():
            # GO Text Analysis

              # render the html page
        print (myText.getMostCommonWords(10))
        return render_template('results.html',
                           title='Text Analysis',
                           inputTypeText = typeText,
                           originalText = fragment,
                           numChars = myText.length(),
                           numSentences = myText.getSentences(),
                           numTokens = myText.getTokens(),
                           uniqueTokens = uniqueTokensText,
                           commonWords = myText.getMostCommonWords(10),
                           normalization=myText.getNormalization())

    elif 'SA'  in request.form.values():
        
         return render_template('sentimentAnalysis.html',
                           title='Sentiment Analysis',
                           inputTypeText = typeText,
                           originalText = fragment,
                           numChars = myText.length(),
                           numSentences = myText.getSentences(),
                           numTokens = myText.getTokens(),
                           uniqueTokens = uniqueTokensText,
                           OR = myText.getPositiveORNegative(),
                           PositiveORNegative=myText.tokenise(),
                           And=myText.getPositiveORNegative2()

         )

    elif 'DA'  in request.form.values():
         davalue = [('jed',0.58),('ruh',0.31)]
         myText = TextAnalyser(userText, language) # new object
         return render_template('dialect.html',
                           title='Dialect Model',
                           inputTypeText = typeText,
                           originalText = fragment,
                           numChars = myText.length(),
                           numSentences = myText.getSentences(),
                           numTokens = myText.getTokens(),
                           uniqueTokens = uniqueTokensText,
                           topCities = myText.getMostCommonCities(),
                           topRegions = myText.getMostCommonRegion(),
                           topCountries = myText.getMostCommonCountry()
         )

    

  # render web form page
@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def initial():
      # render the initial main page
    return render_template('index.html',
                           title = 'NLP3o - Your input',
                           form = InputTextForm())

@app.route('/results')
def results():
    return render_template('index.html',
                           title='NLP3o - Text Analysis')

  # render about page
@app.route('/about')
def about():
    return render_template('about.html',
                           title='About NLP3o')
