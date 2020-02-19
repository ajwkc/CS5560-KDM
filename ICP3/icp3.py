import json
from nltk.corpus import wordnet as wn
from stanfordcorenlp import StanfordCoreNLP


def main():
    # Set up the local server
    nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)

    # Define a sentence for processing
    sentence = 'Bill likes to eat vanilla ice cream with chocolate chips.'

    # Define the annotation tags
    output = nlp.annotate(sentence, properties={
        'annotators': 'tokenize, ssplit, pos, depparse, natlog, openie',
        'outputFormat': 'json',
        'openie.triple.strict': 'true',
        'openie.max_entailments_per_clause': '1'
    })

    # Print the JSON info in readable format
    a = json.loads(output)
    print('The subject, object and predicate of the given sentence are:\n')
    print(a['sentences'][0]['openie'], '\n')
    result = [a['sentences'][0]['openie'] for item in a]
    for i in result:
        for relation in i:
            relationSent = relation['relation'], relation['subject'], relation['object']
            print('The triplet of the given sentence is:\n')
            print(relationSent, "\n\n")

    # WordNet functionality
    word = wn.synsets("worm")

    # Print the word
    print('The word is:' ,word[0].lemmas()[0].name(), '\n')

    # Definition of the word:
    print('The definition is: ', word[0].definition(), '\n')

    # Examples of the word in use in sentences:
    print('Examples of the word:', word[0].examples(), '\n')

    # synonyms and antonyms using wordnet using word
    synonyms = []
    antonyms = []

    for syn in wn.synsets("good"):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    print('The synonyms of good are: ')
    print(set(synonyms))
    print('\n')
    print('The antonyms of good are: ')
    print(set(antonyms))
    print('\n')


    print('Set of hyponyms:\n', word[0].hyponyms(), '\n')
    print('Set of hypernyms:\n', word[0].hypernyms(), '\n')
    print('Set of part-meronyms:\n', word[0].part_meronyms(), '\n')
    print('Set of substance-meronyms:\n', word[0].substance_meronyms(), '\n')
    print('Set of member-holonyms:\n', word[0].member_holonyms(), '\n')
    print('Set of part-meronyms:\n', word[0].part_meronyms(), '\n')

    print('Entailments of the word Breathe:\n', wn.synset('breathe.v.01').entailments())


if __name__ == '__main__':
    main()
