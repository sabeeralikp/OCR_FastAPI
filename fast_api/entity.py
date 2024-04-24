import spacy
import spacy_transformers

nlp = spacy.load("model-best")


def extract_entity(text, entities):
    doc = nlp(text)
    for entity in doc.ents:
        entities[entity.label_].add(entity.text)
    return entities
