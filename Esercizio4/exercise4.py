import re
import sys

import numpy as np
from numpy import mean
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
import os
from pip._vendor import requests

path = os.path.dirname(__file__) + '/'


def get_synset_terms(sense): #retrieve synset_terms of a given BN_synsetID (sense)
    """
    It use the BabelNet HTTP API for getting the first three Lemmas of the word
    associated to the given Babel Synset ID.
    :param sense: sense's BabelID
    :return: the first three lemmas of the given sense. An error string if
    there are none
    """

    url = "https://babelnet.io/v5/getSynset"
    params = {
        "id": sense,
        #"key": "67adc825-bbcc-4cd4-8d6c-71ecdb875e7c",   #alternative key: 67adc825-bbcc-4cd4-8d6c-71ecdb875e7c
        "key": "0e1b3e64-3e24-412b-a39d-7e5b2536ed9f",  # my API key        
        "targetLang": "IT"  # Important: we are searching results in italian
    }

    req = requests.get(url=url, params=params)
    data = req.json()
    #print("sensi")
    #print(data["senses"])

    synset_terms = []

    i = 0  # used to loop over the first three synset terms ("fullLemma")
    j = 0  # used to loop over all the senses in synset
    while j < len(data["senses"]) and i < 3:
        term = data["senses"][j]["properties"]["fullLemma"]

        # added some preprocess
        term = re.sub('\_', ' ', term).lower() #remove '_' from multiwords fullLemmas

        if term not in synset_terms:
            synset_terms.append(term)
            i += 1

        j += 1

    if len(synset_terms) == 0:
        return "Empty synset terms"
    else:
        return synset_terms


def parse_nasari():
    """
    It parses the NASARI's Embedded input.
    :return: First: a dictionary in which each BabelID is associated with the
    corresponding NASARI's vector. Second: a lexical dictionary that associate
    to each BabelID the corresponding english term.

    {babelId: [nasari vector's values]}, {babelID: word_en}
    """

    nasari = {}
    babel_word_nasari = {}

    with open(path + config["nasari"], 'r', encoding="utf8") as file:
        for line in file.readlines():
            lineSplitted = line.split()  # default sep (split at any whitespace)
            k = lineSplitted[0].split("__") # separate bn_synset (k[0]) from eng_word (k[1])
            babel_word_nasari[k[0]] = k[1]  # {babelID: word_en}
            lineSplitted[0] = k[0]  # storing bn_synset value in lineSplitted[0], deleting the eng_word associated
            i = 1
            values = []
            while i < len(lineSplitted):
                values.append(float(lineSplitted[i])) # append in "values" list all the nasari vector's value
                                                      # associated to each bn_sysnet (lineSplitted[0])
                i += 1
            nasari[lineSplitted[0]] = values # {babelId: [nasari vector's values]}

    return nasari, babel_word_nasari  # {1} {2}


def parse_italian_synset():
    """
    It parses SemEvalIT17 file. Each italian term is associated with a list of BabelID.
    :return: a dictionary containing the italian word followed by the list of its BabelID.
    :Format: {word_it: [BabelID]}
    """

    sem_dict = {} # key: it_words, values: "synseys" list associated
    synsets = []  # values of {sem_dict}
    term = "first_step"  # only for the first time
    with open(path + config["input_italian_synset"], 'r', encoding="utf8") as file:
        for line in file.readlines():
            line = line[:-1].lower() # deletes final '\n' character and .lower
            if "#" in line: # if i have encountered a new it_word (and synsets of the previous it_word are finished)
                line = line[1:] # deletes first '#' character
                if term != "first_step":  # only for the first time
                    sem_dict[term] = synsets # assign values ([synsets]) to each dict key (term) -> (the previous #key parsed)
                term = line # prepare dict key for next "for iteration" (to start to fill its values)
                synsets = [] # restore synsets list (for the new dict key to consider)
            else:
                synsets.append(line) # fill "synsets" list 
                                     # (= values for the key "term" set at previous "for iteration")
    return sem_dict


def parse_word(path):
    """
    it parses the annotated words's file.
    :param path to the annotated word's file.
    :return: list of annotated terms. Format: [( (w1, w2), value )]
    """

    annotation_list = [] # tuple list [ ( (w1,w2), value ) , ( (x,y), v ), (...) ]
    with open(path, 'r', encoding="utf-8-sig") as file:
        for line in file.readlines():
            splitted_line = line.split("\t") # split per "TAB" (sep)
            copule_words = (splitted_line[0].lower(), splitted_line[1].lower()) # (w1,w2)
            value = splitted_line[2].replace("\n", "") # value
            annotation_list.append((copule_words, float(value))) # append each ( (w1,w2), value ) to the list
    return annotation_list


def parse_sense(path):
    """
    it parses the senses in the file.
    :param path to the senses file.
    :return: list of annotated senses and associated terms. Format: [(t1, t2, s1, s2)]
    """

    sense_list = []
    with open(path, 'r', encoding="utf-8-sig") as file:
        for line in file.readlines():
            #print(line)
            splitted_line = line.split("\t")
            couple_word = (splitted_line[0].lower(), splitted_line[1].lower()) #(t1, t2)
            copule_sense = (splitted_line[2].lower(), splitted_line[3].lower().replace("\n","")) #(s1,s2)
            sense_list.append((couple_word[0], couple_word[1], copule_sense[0], copule_sense[1]))
                             #  append each ( t1, t2, s1, s2 ) to the list

    return sense_list


def similarity_vector(babel_id_word1, babel_id_word2, nasari_dict):
    """
    It computes the cosine similarity between the two given NASARI vectors
    (with Embedded representation).
    :param babel_id_word1: list of BabelID of the first word (from {SemEval17[word1]})
    :param babel_id_word2: list of BabelID of the second word (from {SemEval17[word2]})
    :param nasari_dict: NASARI dictionary (from mini_NASARI parsed -> {babelId: [nasari vector's values]})
    :return: the couple of senses (their BabelID) that maximise the score and
    the cosine similarity score itself.
    """

    max_value = 0
    senses = (None, None)
    for bid1 in babel_id_word1: # for all BN synset of w1...
        for bid2 in babel_id_word2: # ... and w2
            if bid1 in nasari_dict.keys() and bid2 in nasari_dict.keys(): # if those BN synsets are present in mini_NASARI keys
                                                                          # ( {babelId: [nasari vector's values]} )
                # Storing the NASARI values of bid1 and bid2
                v1 = nasari_dict[bid1]
                v2 = nasari_dict[bid2]

                # Transforming the V1 and V2 array into a np.array (numpy array).
                # Array dimensions: 1 x len(v).
                n1 = np.array(v1).reshape(1, len(v1))
                n2 = np.array(v2).reshape(1, len(v2))

                # Computing and storing the cosine similarity (between nasari vector's values of bid1 and bid2).
                t = cosine_similarity(n1, n2)[0][0] #extract from the kernel matrix the CosSim value
                if t > max_value:
                    max_value = t # max CosSim value found at this time
                    senses = (bid1, bid2) # best senses_couple / BabelID found at this time
    return senses, max_value # ( ((best_bid1, best_bid2), max_value )


def evaluate_correlation_level(v1, v2):
    """
    It evaluates the correlation between the system annotations (v2) and the
    human annotations (v1) using Pearson and Spearman  metrics.
    :param v1: the first annotated vector (human annotations)
    :param v2: the second annotated vector (algorithm annotations)
    :return: Pearson and Spearman indexes
    """

    return pearsonr(v1, v2)[0], spearmanr(v1, v2)[0] # take P e S correlation coefficients,
                                                     # given the 2 vector annotations


global config  # Dictionary containing all the script settings. Used everywhere.

if __name__ == "__main__":

    config = {
        "input_gold_1": "input/mydata/gold_1.txt",
        "input_gold_2": "input/mydata/gold_2.txt",
        "input_gold_3": "input/mydata/gold_3.txt",
        "input_senses_1": "input/mydata/my_senses_1.tsv",
        "input_senses_2": "input/mydata/my_senses_2.tsv",
        "input_senses_3": "input/mydata/my_senses_3.tsv",
        "input_senses": "input/mydata/my_senses.tsv",
        "input_italian_synset": "input/SemEval17_IT_senses2synsets.txt",
        "nasari": "input/mini_NASARI.tsv",
        "output": "output/"
    }

    nasari_dict, babel_word_nasari = parse_nasari()
    italian_senses_dict = parse_italian_synset()

    # Task 1: Semantic Similarity
    #
    # 1. annotate by hand the couple of words in [0,4] range
    # 2. compute inter-rate agreement with Spearman and Pearson indexes
    # 3. Compute the cosine similarity between the hand-annotated scores and
    # Nasari best score given the two terms
    # 4. Evaluate the total quality using again the Spearman and Pearson
    # indexes between the human annotation scores and the Nasari scores.

    print('Task 1: Semantic Similarity')

    annotations_1 = parse_word(path + config["input_gold_1"])
    annotations_2 = parse_word(path + config["input_gold_2"])
    annotations_3 = parse_word(path + config["input_gold_3"])

    # 1. Annotation's scores, used for evaluation
    scores_human_1 = list(zip(*annotations_1))[1] # take all values in the gold annotation (after separating all (w1,w2) from all (values))
    scores_human_2 = list(zip(*annotations_2))[1]
    scores_human_3 = list(zip(*annotations_3))[1]


    # Computing the mean value for each couple of annotation scores (from 2 different annotators)
    scores_human_mean_1 = [(x + y) / 2 for x, y in zip(scores_human_1, scores_human_2)]
    print('\tMean value: {0:.2f}'.format(mean(scores_human_mean_1)))
    scores_human_mean_2 = [(x + y) / 2 for x, y in zip(scores_human_1, scores_human_3)]
    print('\tMean value: {0:.2f}'.format(mean(scores_human_mean_2)))
    scores_human_mean_3 = [(x + y) / 2 for x, y in zip(scores_human_2, scores_human_3)]
    print('\tMean value: {0:.2f}'.format(mean(scores_human_mean_3)))
    scores_human_mean = [(x + y + z) / 3 for x, y, z in zip(scores_human_1, scores_human_2, scores_human_3)]

    # 2. Computing the inter-rate agreement. This express if the two annotations are consistent
    inter_rate_pearson_1, inter_rate_spearman_1 = evaluate_correlation_level(scores_human_1, scores_human_2)
    print('\tInter-rate agreement - Pearson: {0:.2f}, Spearman: {1:.2f}'
          .format(inter_rate_pearson_1, inter_rate_spearman_1))
    inter_rate_pearson_2, inter_rate_spearman_2 = evaluate_correlation_level(scores_human_1, scores_human_3)
    print('\tInter-rate agreement - Pearson: {0:.2f}, Spearman: {1:.2f}'
          .format(inter_rate_pearson_2, inter_rate_spearman_2))
    inter_rate_pearson_3, inter_rate_spearman_3 = evaluate_correlation_level(scores_human_2, scores_human_3)
    print('\tInter-rate agreement - Pearson: {0:.2f}, Spearman: {1:.2f}'
          .format(inter_rate_pearson_3, inter_rate_spearman_3))

    inter_rate_pearson_mean = (inter_rate_pearson_1 + inter_rate_pearson_2 + inter_rate_pearson_3)/3
    inter_rate_spearman_mean = (inter_rate_spearman_1 + inter_rate_spearman_2 + inter_rate_spearman_3)/3
    print('\tInter-rate (mean) agreement - Pearson: {0:.2f}, Spearman: {1:.2f}'
          .format(inter_rate_pearson_mean, inter_rate_spearman_mean))

    # 3. Computing the cosine similarity between the hand-annotated scores and
    # Nasari best score given the two terms
    with open(path + config["output"] + 'results_task1.tsv', "w", encoding="utf-8") as out:
        annotations_algorithm = []
        for couple in annotations_1:  # is equal to use annotations_1 or annotations_2, because the words are the same
            key = couple[0] # (t1, t2) couple of words 
            (s1, s2), score = similarity_vector(italian_senses_dict[key[0]], italian_senses_dict[key[1]], nasari_dict)
            annotations_algorithm.append(((s1, s2), score*4)) # storing each result in list
            out.write("{0}\t{1}\t{2:.2f}\n".format(key[0], key[1], score*4)) # writing the out file's lines

    scores_algorithm = list(zip(*annotations_algorithm))[1] # list of all the scores found with CosSim, 
                                                            # for each couple of words in input

    # 4. Evaluate the total quality using again the Spearman and Pearson
    # indexes between the human annotation scores and the Nasari scores.
    pearson, spearman = evaluate_correlation_level(scores_human_mean, scores_algorithm)
    print('\tEvaluation - Person: {0:.2f}, Spearman: {1:.2f}'.format(pearson, spearman))

    # ------------------------------------------------------------------------------------------------------------------

    print("\nTask 2: Sense Identification.")
    # Task 2: Sense Identification
    #
    # 1. annotate by hand the couple of words in the format specified in the README
    # 2. compute inter-rate agreement with the Cohen's Kappa score
    # 3. Compute the cosine similarity between the hand-annotated scores and
    # Nasari best score given the two terms
    # 4. Evaluate the total quality using the argmax function. Evaluate both
    # the single sense and both the senses in the couple.

    senses_1 = parse_sense(path + "input/mydata/my_senses_1.tsv")
    senses_2 = parse_sense(path + "input/mydata/my_senses_2.tsv")
    senses_3 = parse_sense(path + "input/mydata/my_senses_3.tsv")

    # score delle coppie annotate di sensi uniti in un unica lista(di coppie di sensi annotati), una per ogni annotatore
    score_senses_1_couple = []
    score_senses_2_couple = []
    score_senses_3_couple = []

    scores_senses_1_first = list(zip(*senses_1))[2] # taking 3^ column (w1_Sensevalue) of the input annotated file
    score_senses_1_second = list(zip(*senses_1))[3] # taking 4^ column (w2_Sensevalue) of the input annotated file
    for couple_s in zip(scores_senses_1_first, score_senses_1_second): # taking each couple of scores (w1_value, w2_value)
        #print("coppia annotata di sensi")
        #print(couple_s)
        score_senses_1_couple.append(couple_s) #append each time a couple of annotated BN synsets (ex: terremoto_Sensevalue, scossa_Sensevalue)
    #print(score_senses_1_couple)
    
    score_senses_2_couple = []
    scores_senses_2_first = list(zip(*senses_1))[2]
    score_senses_2_second = list(zip(*senses_1))[3]
    for couple_s in zip(scores_senses_2_first, score_senses_2_second):
        #print("coppia annotata di sensi")
        #print(couple_s)
        score_senses_2_couple.append(couple_s)
    
    score_senses_3_couple = []
    scores_senses_3_first = list(zip(*senses_3))[2]
    score_senses_3_second = list(zip(*senses_3))[3]
    for couple_s in zip(scores_senses_3_first, score_senses_3_second):
        #print("coppia annotata di sensi")
        #print(couple_s)
        score_senses_3_couple.append(couple_s)

    # 2. Computing the inter-rate agreement. This express if the two score are consistent
    k_1 = 0
    for c in zip(score_senses_1_couple, score_senses_2_couple): # accoppio le 2 diverse annotazioni(coppie di sensi annotati) dei 2 annotatori,
                                                                # una coppia di coppie alla volta, e calcolo k-cohen (inter-agreement)
        print("coppia di 2 diverse annotazioni per una coppia di termini, da valutare con k-cohen")
        print(c)
        k_1 += (cohen_kappa_score(list(c[0]), list (c[1])))
    k_2 = 0
    for c in zip(score_senses_2_couple, score_senses_3_couple):
        k_2 += (cohen_kappa_score(list(c[0]), list (c[1])))
    k_3 = 0
    for c in zip(score_senses_1_couple, score_senses_3_couple):
        k_3 += (cohen_kappa_score(list(c[0]), list (c[1])))
    
    k_mean = ( k_1 + k_2 + k_3 ) / 3
    print('\tInter-rate mean agreement - Cohen Kappa: {0:.2f}'.format(k_mean))


    with open(path + "output/" + "results_task_2.tsv", "w", encoding="utf-8") as out:

        i = 0  # used for print progress bar
        first_print = True  # used for print progress bar

        # used for final comparison. It is a in-memory copy of the output file
        nasari_out = []
        for row in senses_1: #uso i BNsynset annotati di Maltese (senses_1)

            # In this case I re-use the similarity_vector function, which use
            # the cosine similarity to compute again the two senses that
            # produce the maximal similarity score. The "score" variable is
            # unused, so it's substituted by the don't care variable "_".
            (s1, s2), _ = similarity_vector(italian_senses_dict[row[0]], italian_senses_dict[row[1]], nasari_dict)

            # if both Babel Synset exists and are not None
            if s1 is not None and s2 is not None:
                out.write("{}\t{}\t{}\t{}\t".format(row[0], row[1], s1, s2)) # writing the out file's lines

                out_terms_1 = get_synset_terms(s1)
                out_terms_2 = get_synset_terms(s2)
                nasari_terms_1 = ""
                nasari_terms_2 = ""

                for t1 in out_terms_1: # collecting the first 3 synset terms (fullLemmas) of the given sense (the best sense found with CosSim)
                    if t1 != out_terms_1[len(out_terms_1) - 1]:
                        out.write(t1 + ",")  # if not the last term, put a "," after t1 (synset term retrieved)
                        nasari_terms_1 += t1 + ","
                    else:
                        out.write(t1 + "\t")  # otherwise put a separator after t1 (synset term retrieved)
                        nasari_terms_1 += t1

                for t2 in out_terms_2:
                    if t2 != out_terms_2[len(out_terms_2) - 1]:
                        out.write(t2 + ",")  # if not the last term, put a ","
                        nasari_terms_2 += t2 + ","
                    else:
                        out.write(t2 + "\n")  # otherwise put a separator
                        nasari_terms_2 += t2
            else:
                out.write("{}\t{}\tNone\tNone\tNone\tNone\n".format(row[0], row[1]))

            # updating percentage
            i += 1

            if first_print:
                print('\tDownloading terms from BabelNet.')
                print('\t#', end="")
                first_print = False
            if i % 10 == 0:
                print('#', end="")
            else:
                print('-', end="")

            # populate the nasari_out list.
            nasari_out.append((row[0], row[1], s1, s2, nasari_terms_1, nasari_terms_2))

        count_single = 0
        count_couple = 0
        for sense_row in senses_1: #score annotati da Maltese (senses_1)
            for nasari_row in nasari_out: #score calcolati da nasari nel codice
                arg0 = sense_row[2] == nasari_row[2] #1^ senso uguale
                arg1 = sense_row[3] == nasari_row[3] #2^ senso uguale
                if arg0:
                    count_single += 1
                if arg1:
                    count_single += 1
                if arg0 and arg1:
                    count_couple += 1
        print("\n\tSingle: {0} / 100 ({0}%) - Couple: {1} / 50 ({2:.0f}%)"
              .format(count_single, count_couple, (count_couple * 100 / 50)))
              