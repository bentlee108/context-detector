"""
=========================================================================================
Author: Brent Anderson
Date created:   05/08/2022
Description: Program to detect context of a given statement and output smart home action.
=========================================================================================
"""
import speech_recognition as sr
from nltk.corpus import stopwords
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# from textblob import Word
# from textblob import TextBlob
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from nltk.corpus import wordnet as wn
# import nltk
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# Initialize the recognizer and mic globally
r = sr.Recognizer()
mic = sr.Microphone(device_index=1)

def main():         
    # use the microphone as source for input.
    with mic as source:

        if not isinstance(r, sr.Recognizer): raise TypeError("`recognizer` must be `Recognizer` instance")

        if not isinstance(mic, sr.Microphone): raise TypeError("`microphone` must be `Microphone` instance")

        r.adjust_for_ambient_noise(source)
            
        #listens for the user's input
        print("\n\nSpeak now:")
        audio = r.listen(source)
        print("Stop speaking.\n\n")
            
        # Using google to recognize audio
        text = r.recognize_google(audio, language="en-US")

        # Text bank before obtaining context
        # text = "My house is so dirty, I've really got to clean it, but I just can't seem to find the motivation to do it." # Run vacuum
        # text = "I'm so cold in this house, I can't take it anymore! I'm gonna go take a nice hot shower to warm up!" # Turn down A/C
        # text = "I can't remember the last time I vaccumed. I can't see the hard wood floors anymore under all of this dust!" # Run vacuum
        # text = "I'm cold. So very cold. It's too cold in here. I don't know how much longer before I turn into a popsicle" # Turn down A/C
        # text = "Hello, how are you today? I can't beleive this weather we're having, can you beleive this? It's so crazy! Ugh, I can see all of the dust in my house because of all of this sun!" # Run vacuum

        # Convert text to list and remove stopwords (if not completely comprised of stopwords)
        data = text.replace('"', "").split(" ")
        stopWords = set(stopwords.words('english'))
        temp = [w for w in data if w not in stopWords and len(w) > 2]
        if temp: data = temp

        # tokenize and vectorize the data
        # TFIDF = (occurences of word in document / total number of words in document) * (log[total # of documents in corpus / number of documents containing word])
        vectorizer = TfidfVectorizer(max_features=500)
        vectorData = vectorizer.fit_transform(data)

        # Non-Negative Matrix Factorization model...
        '''
        Given a matrix M x N, where M = Total number of documents and N = total number of words,
        NMF is the matrix decompostition that generates the Features with M rows and K columns,
        where K = total number of topics and the Components matrix is the matrix of K by N.
        The Product of the Features and Components matricies results in the approximation of the TF-IDF.
        '''
        nmf_model = NMF(n_components=1, init='random', random_state=0)
        nmf_model.fit_transform(vectorData)

        # get the feature names
        featureNames = vectorizer.get_feature_names_out()

        # Sort feature names and preprocess for model use.
        topicsAndComponents = zip(featureNames, nmf_model.components_[0])
        sortedTopics = sorted(topicsAndComponents, key=lambda x:x[1], reverse=True)[:20]
        topics = [topic for (topic, comp) in sortedTopics]
        topics = [' '.join(topics)]

        # Create text for model training
        # This is also what the text bank looks like after obtaining context
        text_data = np.array([
            'beleive dust having this crazy hello house today see can weather re we it sun ugh',
             'take gonna hot anymore cold house nice up can shower warm',
              'it find got can dirty house seem clean motivation ve really',
               'cold know longer it much popsicle here turn',
                'can floors hard anymore dust last remember vaccumed see wood time',
              ])
        
        # Create bag of words
        count = CountVectorizer(stop_words = None)
        bag_of_words = count.fit_transform(text_data)
        
        # Quick data match for text_data to topics excluding first array value.
        temp_text_data = np.delete(text_data, 0)
        temp_text_data = np.insert(temp_text_data, 0, topics[0])
        topics = temp_text_data

        # Create bag of words for prediction
        predict_bag_of_words = count.fit_transform(topics)

        # Create feature matrix
        features = bag_of_words.toarray()
        predict_features = predict_bag_of_words.toarray()
        predict_features = list(predict_features[0])

        # Pad if necessary
        if (len(predict_features) < len(features[0])):
            for i in range(len(features[0]) - len(predict_features)): 
                predict_features.append(0)

        # Create target vector
        target = np.array([0, 1, 0, 1, 0])

        # Create multinomial naive Bayes object with prior probabilities of each class
        # The multinomial distribution normally requires integer feature counts. 
        # However, in practice, fractional counts such as tf-idf may also work.
        classifer = MultinomialNB(class_prior=[0.25, 0.5])

        # Train model
        model = classifer.fit(features, target)

        # Predict new observation's class and output smart home action
        prediction = model.predict([predict_features])
        smartHomeAction = "Run autonomous vacuum." if prediction == 0  else "Turn air conditioning down."
        print("\nSmart home action: ", smartHomeAction)
        print("\n")

if __name__ == "__main__":
    main()