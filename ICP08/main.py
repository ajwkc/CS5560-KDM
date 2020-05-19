import spacy
import textacy

f = open('abs3.txt', 'r')
nlp = spacy.load("en_core_web_sm")
doc = nlp(f.read())
f.close()

for token in doc:
    print(token.text, token.pos_, token.dep_)

tuples_list = []
tuples = textacy.extract.subject_verb_object_triples(doc)
tuples_to_list = list(tuples)
if tuples_to_list != []:
    tuples_list.append(tuples_to_list)

print(tuples_list)