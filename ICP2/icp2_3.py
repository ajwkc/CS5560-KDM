#!/usr/bin/env python

from pycorenlp import StanfordCoreNLP

# StanfordCoreNLP's Python wrapper doesn't handle sentiment analysis


def main():

    # Java server configuration
    nlp = StanfordCoreNLP('http://localhost:9000')

    # Annotate the text file for sentiment analysis
    file = open('walden.txt')
    text = file.read()
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize, ssplit, sentiment',
        'outputFormat': 'json'
    })

    # Print each sentence, its index, and its sentiment score
    for sentence in output["sentences"]:
        print("%d: '%s': %s %s" % (
            sentence["index"],
            " ".join([token["word"] for token in sentence["tokens"]]),
            sentence["sentimentValue"], sentence["sentiment"]))

    # Close the file
    file.close()


if __name__ == '__main__':
    main()