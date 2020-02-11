from pycorenlp import StanfordCoreNLP


def main():
    nlp = StanfordCoreNLP('http://localhost:9000')

    file = open('walden.txt')
    text = file.read()
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize, ssplit, sentiment',
        'outputFormat': 'json'
    })

    for sentence in output["sentences"]:
        print("%d: '%s': %s %s" % (
            sentence["index"],
            " ".join([token["word"] for token in sentence["tokens"]]),
            sentence["sentimentValue"], sentence["sentiment"]))

    file.close()


if __name__ == '__main__':
    main()