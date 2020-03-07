from stanfordcorenlp import StanfordCoreNLP


def main():

    # Java server configuration
    host = 'http://localhost'
    port = 9000
    nlp = StanfordCoreNLP(host, port=port, timeout=30000)

    # Sample sentence for parsing
    sentence = 'Susan gave me a sandwich, but I did not eat it.'

    # Part of Speech Tagging
    print('POS：', nlp.pos_tag(sentence))

    # Tokenization
    print('Tokenize：', nlp.word_tokenize(sentence))

    # Named Entity Recognition
    print('NER：', nlp.ner(sentence))

    # Parser
    print('Parser：')
    print(nlp.parse(sentence))
    print(nlp.dependency_parse(sentence))

    # Co-reference Resolution
    print('Co-references:')
    print(nlp.coref(sentence))

    # Close the parser
    nlp.close()


if __name__ == '__main__':
    main()
