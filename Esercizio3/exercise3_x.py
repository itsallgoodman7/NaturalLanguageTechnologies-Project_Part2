import sys
import os
from pathlib import Path

from tqdm import tqdm

from utilities import get_title_topic, create_context,aux_bag_of_word
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math



path = os.path.dirname(__file__) + '/'


def compute_overlap(topic, paragraph):
    """
    Support function used in Weighted Overlap's function below.
    :param topic: Vector representation of the topic
    :param paragraph: Vector representation of the paragraph
    :return: intesection between the given parameters
    """

    return topic & paragraph


def rank(vector, nasari_vector):
    """
    Computes the rank of the given vector.
    :param vector: input vector
    :param nasari_vector: input Nasari vector
    :return: vector's rank (position inside the nasari_vector)
    """

    for i in range(len(nasari_vector)):
        if nasari_vector[i] == vector:
            return i + 1


def weighted_overlap(topic_nasari_vector, paragraph_nasari_vector):
    """
    Implementation of the Weighted Overlap metrics (Pilehvar et al.)
    :param topic_nasari_vector: Nasari vector representing the topic
    :param paragraph_nasari_vector: Nasari vector representing the paragraph
    :return: square-rooted Weighted Overlap if exist, 0 otherwise.
    """

    overlap_keys = compute_overlap(topic_nasari_vector.keys(),
                                   paragraph_nasari_vector.keys())
    #print(topic_nasari_vector)
    #print(paragraph_nasari_vector)

    overlaps = list(overlap_keys)

    if len(overlaps) > 0:
        # sum 1/(rank() + rank())
        den = sum(1 / (rank(q, list(topic_nasari_vector)) +
                       rank(q, list(paragraph_nasari_vector))) for q in overlaps)

        # sum 1/(2*i)
        num = 0
        for i in range(1,len(overlaps)+1):
            num += 1/ (2 * i)

        return den / num

    return 0


def parse_nasari_dictionary():
    """
    It parse the Nasari input file, and it converts into a more convenient
    Python dictionary.
    :return: a dictionary representing the Nasari input file. Fomat: {word: {term:score}}
    """

    nasari_dict = {}
    with open(path + config["nasari"], 'r', encoding="utf8") as file:
        for line in file.readlines():
            splits = line.split(";")
            vector_dict = {}

            for term in splits[2:]:
                k = term.split("_")
                if len(k) > 1:
                    vector_dict[k[0]] = k[1]            # vector[term] = score

            nasari_dict[splits[1].lower()] = vector_dict

    return nasari_dict


def summarization(document, nasari_dict, percentage):
    """
    Applies summarization to the given document, with the given percentage.
    :param document: the input document
    :param nasari_dict: Nasari dictionary
    :param percentage: reduction percentage
    :return: the summarization of the given document.
    """

    # getting the topics based on the document's title.
    topics = get_title_topic(document, nasari_dict)

    paragraphs = []
    i = 0
    # for each paragraph, except the title (document[0])
    for paragraph in document[1:]:
        context = create_context(paragraph, nasari_dict)

        paragraph_wo = 0  # Weighted Overlap average inside the paragraph.

        for word in context:
            # Computing WO for each word inside the paragraph.
            topic_wo = 0
            for vector in topics:
                topic_wo = topic_wo + weighted_overlap(vector, word)
            if topic_wo != 0:
                topic_wo = topic_wo / len(topics)

            # Sum all words WO in the paragraph's WO
            paragraph_wo += topic_wo

        if len(context) > 0:
            paragraph_wo = paragraph_wo / len(context)
            # append in paragraphs a tuple with the index of the paragraph (to
            # preserve order), the WO of the paragraph and the paragraph's text.
            paragraphs.append((i, paragraph_wo, paragraph))
        i += 1

    to_keep = len(paragraphs) - int(round((percentage / 100) * len(paragraphs), 0))

    # Sort by highest score and keeps all the important entries. From first to "to_keep"
    new_document = sorted(paragraphs, key=lambda x: x[1], reverse=True)[:to_keep]
    # Restore the original order.
    new_document = sorted(new_document, key=lambda x: x[0], reverse=False)
    # delete unnecessary fields (x[0] which contains the "i" and x[1] which
    # contains the WO of the paragraph) inside new_document in order to
    # keep oly the text (I associated to each paragraph a score based on the
    # "importance").
    new_document = list(map(lambda x: x[2], new_document))

    new_document = [document[0]] + new_document
    return new_document


def parse_document(file):
    """
    It parse the given document.
    :param file: input document
    :return: a list of all document's paragraph.
    """

    document = []
    data = file.read_text(encoding='utf-8')
    lines = data.split('\n')

    for line in lines:
        # If the "#" character is present, it means the line contains the
        # document original link. So, if the # is not present,
        # we have a normal paragraph to append to the list.
        if line != '' and '#' not in line:
            line = line[:-1]  # deletes the final "\n" character.
            document.append(line)

    return document

global config  # Dictionary containing all the script settings. Used everywhere.

if __name__ == "__main__":

    

    config = {
        "input": "input/text-documents",
        "nasari": "input/dd-small-nasari-15.txt",
        "output": "output/",
        "percentage": 30
    }

    print("Summarization.\nReduction percentage: {}".format(config["percentage"]))

    nasari_dict = parse_nasari_dictionary()

    # Inspecting the input files
    path_to_input = Path(path + config["input"])
    files_documents = list(path_to_input.glob('./*.txt'))

    count = 0 # n of documents
    total_blue_score = 0
    total_rouge_score = 0

    for file in files_documents:
        count += 1
        document = parse_document(file)

        # For each document, do some pretty print of the original.
        # It will be used later for comparison.
        with open(path + config["output"] + 'Original-' + file.name,
                  'w', encoding='utf-8') as out_original:
            for paragraph in document:
                out_original.write(paragraph + '\n')

        # For each document do summarization.
        sum_document = summarization(document, nasari_dict, config["percentage"])

        with open(path + config["output"] + str(config["percentage"]) + '-' + file.name,
                  'w', encoding='utf-8') as out_summarized:
            for paragraph in sum_document:
                out_summarized.write(paragraph + '\n')


        #### Evaluation #####

        docs = []
        for paragraph in document:
            docs = docs + list(aux_bag_of_word(paragraph))
        docs = ' '.join(docs)
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([docs])
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()

        df = pd.DataFrame(denselist, columns=feature_names)
        df = df.to_dict('list')    

        mvp_words_tfidf = sorted(df.items(), key=lambda x:x[1], reverse=True)[:250]

        # BLEU precision score between gold (tfidf scores) and our algorithm score
        w_list = []
        temp = []
        gold_list = []
        for paragraph in sum_document:
            w_list = w_list + list(aux_bag_of_word(paragraph))
    
            
        for i in range (len(mvp_words_tfidf)):
            gold_list.append(mvp_words_tfidf[i][0])
            #print ("ciao")
            #print (mvp_words_tfidf[i][0])
            
        
        intersection_set = set.intersection(set(w_list), set(gold_list))
        intersection_list = list(intersection_set)
        
        bleu_score = len(intersection_list) / len(w_list)
        print("\nBLEU score of document {0} = : {1:.2f}".format(file.name,bleu_score*100))
        rouge_score = len(intersection_list) / len(gold_list)
        print("ROUGE score of document {0} = : {1:.2f}\n".format(file.name,rouge_score*100))

        total_blue_score += bleu_score
        total_rouge_score += rouge_score


    print("Average BLUE score = {0:.2f}\nAverage ROUGE score = {1:.2f}".format(total_blue_score/count,total_rouge_score/count*100))