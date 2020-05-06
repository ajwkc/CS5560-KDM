import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


def extract_features(word_list):
    return dict([(word, True) for word in word_list])


if __name__ == '__main__':
    # Load positive and negative reviews
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')
    features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                          'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                          'Negative') for f in negative_fileids]
    
    # Split the data into train and test (80/20)
    threshold_factor = 0.45
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))

    features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
    print("\nNumber of training datapoints:", len(features_train))
    print("Number of test datapoints:", len(features_test))

    # Train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(features_train)
    print("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))

    print("\nTop 10 most informative words:")
    for item in classifier.most_informative_features()[:10]:
        print(item[0])

        # Sample input reviews
    input_reviews = [
        '''I can't believe what I just saw! My jaw hung open the entire time, from start to finish. I couldn't 
        feel my face for the entire film. I just shoveled popcorn into my gaping piehole and let the buttery grease 
        cascade down my quivering chin, allowing the glowing screen to wash over me. I was simply transfixed, 
        marveling at the sheer spectacle playing before me. This movie has strengthened my resolve to live.''',

        '''I have not ever seen a movie like this. No other film can accomplish what this one has. Other directors
        may try, but will fail to live up to its grace. Hell, all other directors should call it quits now that
        this movie has blown up the box office. Let's fast track this one straight to DVD, and while we're at
        it, let's shut Hollywood down. They can't top this.''',

        '''I bet this filmmaker thinks he's so clever. Wow! A twist, you say? The bus driver was his uncle all along? 
        Like none of us saw that coming. Look, if you're going to pick up a camera, you at least should learn how to 
        remove the lens cover. That, and keep your fingers out of the shot. Ever heard of film school, champ? Or were 
        you kicked out for prompting the psychosis of your professors? Did you really have to include that scene 
        where the koala bear wins the triathlon? Perhaps this movie would be easier to enjoy if I was drunk.''',
        
        '''I would prefer to surgically remove my eyes than to watch one single minute of this horror. Honestly, 
        I'll take a hot acid bath over listening to its soundtrack. The opening credits alone felt like waves of 
        black widow spiders scuttling up my trousers, as I twitched and convulsed in pain and embarrassment. My 
        nightmares of the past seem pleasant, comforting even, after what I've sat through. I look forward to the day 
        that I may remember what joy was.'''
    ]

    print("\nPredictions:")
    for review in input_reviews:
        print("\nReview:", review)
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        print("Predicted sentiment:", pred_sentiment)
        print("Probability:", round(probdist.prob(pred_sentiment), 2))
