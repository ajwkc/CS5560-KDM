import json
from stanfordcorenlp import StanfordCoreNLP


def main():
    # Set up the local server
    nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)

    # Define a sentence for processing
    sentence = 'Candace likes to eat vanilla ice cream with chocolate chips'

    # Define the annotation tags
    output = nlp.annotate(sentence, properties={
        'annotators': 'tokenize, ssplit,pos,depparse,natlog,openie',
        'outputFormat': 'json',
        'openie.triple.strict': 'true',
        'openie.max_entailments_per_clause':'1'
    })

    # Print the JSON info in readable format
    a = json.loads(output)
    print('The subject, object and predicate of the given sentence are:\n')
    print(a['sentences'][0]['openie'],'\n')
    result = [a['sentences'][0]['openie'] for item in a]
    for i in result:
        for relation in i:
            relationSent = relation['relation'],relation['subject'],relation['object']
            print('The triplet of the given sentence is:\n')
            print(relationSent)

if __name__ == '__main__':
    main()