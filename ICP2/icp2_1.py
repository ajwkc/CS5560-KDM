#!/usr/bin/env python

import nltk
import neuralcoref, spacy


def main():

    # Sample sentences for NLP
    sentence1 = 'The dog saw John in the park.'
    sentence2 = 'The little bear saw the fine fat trout in the rocky brook.'

    # Print the sentences
    print('Sentence1:', sentence1)
    print('Sentence2:', sentence2, '\n')

    # Split up the sentences into their words
    tokens1 = nltk.word_tokenize(sentence1)
    print('Tokens for sentence1:', tokens1)

    tokens2 = nltk.word_tokenize(sentence2)
    print('Tokens for sentence2:', tokens2, '\n')

    # Identify each word's part of speech
    pos1 = nltk.pos_tag(tokens1)
    print('Parts of speech for sentence1:', pos1)

    pos2 = nltk.pos_tag(tokens2)
    print('Parts of speech for sentence2:', pos2, '\n')

    # Identify the named entities (important nouns)
    ner1 = nltk.chunk.ne_chunk(pos1)
    print('Named entities for sentence1:', ner1)

    ner2 = nltk.chunk.ne_chunk(pos2)
    print('Named entities for sentence2:', ner2, '\n')

    # New sentence for demonstrating pronoun co-reference resolution
    sentence3 = "Bob went to the store. He bought his dog a treat."
    print('Sentence3:', sentence3)
    model = spacy.load('en')
    neuralcoref.add_to_pipe(model)
    coref3 = model(sentence3)
    print('Corefs for sentence3: ', coref3._.coref_clusters)


if __name__ == '__main__':
    main()