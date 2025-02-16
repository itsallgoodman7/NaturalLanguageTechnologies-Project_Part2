import re
import csv
import nltk
import os
from nltk.corpus import framenet as fn

from WordNet import WordNet

output = os.path.dirname(__file__) + '/output/'
golden_input = os.path.dirname(__file__) + '/input/gold.csv'


def get_main_clause(frame_name): # disambiguate the "reggente" of a multiword expression (some frame_name like "Religious_belief")
    """
    Get of the main clause from the frame name (in italian "reggente").
    :param frame_name: the name of the frame
    :return: the main clause inside the frame name
    """
    tokens = nltk.word_tokenize(re.sub('\_', ' ', frame_name))
    tokens = nltk.pos_tag(tokens)
    #print("normal order tokens: ")
    #print(tokens)
    #print("reverse tokens: ")
    for elem in tokens: #reversing iteration over pos_tag(tokens) result: [ (w1, POS_tag1), (w2, POS_tag2), (...) ]
                                  # reverse iteration is used to improve the accuracy
        #print(elem)
        if elem[1] == "JJ":  # found in the reverse iteration of tokens
            pass # return word token corresponding to POS_tag NN or NNS (main clause of the multiwprd expression)
        else:
            return elem[0] # VBG, NN(S/P) 

def populate_contexts(frame, mode):
    """
    It populates 2 disambiguation context (one for Framenet and one for Wordnet)
    given a frame name.

    :param frame: the frame name.
    :param mode: a string indicating the way to create context -> the possibility are "Frame name", "FEs" and "LUs".
    :return: a tuple: ( list(ctx_f), dict(ctx_w) ) representing the populated contexts for FN and WN.
    """
    ctx_f = []  # the Framenet context
    ctx_w = {}  # the Wordnet context
    wnac = WordNet()

    if mode == "Frame name":
        # The context in this case contains the frame name and his definition.
        ctx_f = [get_main_clause(frame.name), frame.definition]

        # Here, the context is a dict of synsets associated to the frame name (the given word).
        # In each synset (key of the dict) are usually present (as values) word, glosses and examples.
        ctx_w = wnac.get_disambiguation_context(get_main_clause(frame.name))

    elif mode == "FEs":
        # Populating ctx_f for FEs
        for key in frame.FE: # for all the Frame Elements of the given frame
            ctx_f.append(key) # append FE name
            ctx_f.append(frame.FE[key].definition) # append FE gloss/definition

            # copying all the values inside the ctx_w dictionary.
            temp = wnac.get_disambiguation_context(key) # dict of WN synsets for all FE elements of the given frame as keys
            for k in temp:
                ctx_w[k] = temp[k] # fill context dict with the corresponding info found in temp dict relative to the same key

    elif mode == "LUs":
        # Populating ctx_w for LUs
        for key in frame.lexUnit: # for all the Lexical Units of the given frame
            #print(key)
            lu_key = re.sub('\.[a-z]+', '', key) # regex to keep only the first part of LU's string,
                                                 # cutting the grammatical function specification (.v / .n / ...)
            #print ("key formattata")
            #print(lu_key)
            ctx_f.append(lu_key) # append LU name, cleaned by LU's grammatical function (ex: .v)
            #ctx_f.append(frame.lexUnit[key].definition)

            # copying all the values inside the ctx_w dictionary.
            temp = wnac.get_disambiguation_context(lu_key)
            for k in temp:
                ctx_w[k] = temp[k]

    return ctx_f, ctx_w   # (ctx_f, ctx_w)


def bag_of_words(ctx_fn, ctx_wn):
    """
    Given two disambiguation context, it returns the bag of words mapping
    between the input arguments.
    :param ctx_fn: the first disambiguation context (from Framenet)
    :param ctx_wn: the second disambiguation context (from Wordnet)
    :return: the synset with the highest score
    """
    sentences_fn = set()  # local set of all Framenet names, FEs, LU, and their descriptions
    sentences_wn = {}  # local dictionary of all Wordnet sysnset, glosses and examples.
    ret = None  # temporary return variable
    temp_max = 0  # temporary variable for the score

    for sentence in ctx_fn: # for each FN "context word" in the list 
        for word in sentence.split():
            #print("word_dirty")
            #print(word)
            word_clean = re.sub('[^A-Za-z0-9 ]+', '', word) # Remove all special characters, punctuation and spaces from string
                                                            # regex to match a string of characters that are not a letters or numbers
            #print("word_clean")
            #print(word_clean)
            sentences_fn.add(word_clean)

    # transform the ctx_w dictionary into a set (one for each synset in the WN dict),
    # in order to compute intersection with the FN sentences list
    for key in ctx_wn:
        #print("wn synset")
        #print(key)      # for each WN synset in the dict
        temp_set = set() # one temp_set for each WN synset in the ctx_wn dict
        for sentence in ctx_wn[key]:  # for each sentence inside the WN synset just considered
                                      # (in the for loop of all keys/WNsynsets of the dict)
            if sentence:
                for word in sentence.split():
                    temp_set.add(word)  # add words to temp_set (the temp_set relative to the WN synset just considered)

        # Computing intersection between temp_set (sentences in one WN set, relative to one synset of the WN dict) 
        # and sentences_fn (sentences in the FN set).
        # and then
        # Putting the result inside sentences_wn[key].
        # Each entry in sentences_wn will have the cardinality of the
        # intersection as his "score" at the first position.
        sentences_wn[key] = (len(temp_set.intersection(sentences_fn)), temp_set) # for each WN synset / for each key in ctx_wn
                                                                                 # = tuple (cardinality_intersection_score, set relative to the given WN synset considered)

        # update max score and save the associated sentence.
        if temp_max < sentences_wn[key][0]: # sentences_wn[key][0] = cardinality_intersection_score for that WN synset
            temp_max = sentences_wn[key][0]
            ret = (key, sentences_wn[key]) # setting as the ret value the tuple (WN_synset_considered, tuple (cardinality_intersection_score, set_relative_to_given_WN_synset_considered))

    if ret: # at the end of the for loop, having considered all WN synsets for the WN dict, and compared its sentences with FN sentences,
            # it is returned the first elem of the ret tuple (at its last state / definitive temp_max value) = key -> 
            # the best WN synset with the best sentences (WN context) in context disambiguation with the FN sentences (FN context)
        return ret[0]  # return the synset with the highest (cardinality_intersection_score)

   
def evaluate():
    """
    Doing output evaluation and print the result on the console.
    """
    total_len = 0
    test = 0
    with open(output + 'results.csv', "r", encoding="utf-8") as results:
        with open(golden_input, "r", encoding="utf-8") as golden:
            reader_results = csv.reader(results, delimiter=',') # columns separated by "," in the csv files
            reader_golden = csv.reader(golden, delimiter=',')

            items_in_results = []  # list of items in results
            items_in_golden = []  # list of items in gold

            for line_out in reader_results:
                items_in_results.append(line_out[-1]) # appending to the reader_score_list / gold_score_list the last column of the csv file, 
                                                      # containing the WN synsets chosen by computing the BOW matching between the two FN and WN context

            for line_golden in reader_golden:
                items_in_golden.append(line_golden[-1])

            # counting equal elements
            i = 0
            while i < len(items_in_results):
                if items_in_results[i] == items_in_golden[i]:
                    test += 1 # equal items = equal WN synsets assigned to some element
                i += 1

            total_len = i

    print("\nPrecision: {0} / {1} Synsets -> {2:.2f} %".format(test, total_len, (test / total_len) * 100))


#global config  # Dictionary containing all the script settings. Used everywhere.

if __name__ == "__main__":


    """

        getFrameSetForStudent('Maltese')
        getFrameSetForStudent('Morelli')
        getFrameSetForStudent('Nuzzarello')

        student: Maltese
            ID: 2371	frame: Bond_maturation
            ID:  273	frame: Escaping
            ID:  470	frame: Bearing_arms
            ID: 2243	frame: Commutative_process
            ID: 2790	frame: Locative_scenario

        student: Morelli
            ID: 1603	frame: Scarcity
            ID:  971	frame: Architectural_part
            ID:  317	frame: Releasing
            ID: 1750	frame: Terrorism
            ID: 1025	frame: Connecting_architecture

        student: Nuzzarello
            ID:  516	frame: Preventing_or_letting
            ID:  379	frame: Time_vector
            ID: 1102	frame: People_by_morality
            ID: 1762	frame: Risky_situation
            ID: 1624	frame: Bragging

    """
    #frame_ids = [133, 2980, 405, 1927, 2590]
    frame_ids = [2371, 273, 470, 2243, 2790]    # Maltese frame_ids, from getFrameSetForStudent('Maltese')

    with open(output + 'results.csv', "w", encoding="utf-8") as out:

        print("Assigning Synsets...")

        for frame in frame_ids:
            f = fn.frame_by_id(frame) # obtaining FN frame by its id

            ctx_f, ctx_w = populate_contexts(f, "Frame name")
            sense_name = bag_of_words(ctx_f, ctx_w) # best WN sense for FE name

            out.write("Frame name, {0}, Wordnet Synset, {1}\n".format(f.name, sense_name))

            ctx_f, ctx_w = populate_contexts(f, "FEs")
            i = 0
            while i < len(ctx_f) - 2:
                fe = [ctx_f[i], ctx_f[i + 1]] # adding each time to the fe list FE name + FE definition 
                sense_fes = bag_of_words(fe, ctx_w) # best WN sense for each FEs of each FE
                out.write("Frame elements, {0}, Wordnet Synset, {1}\n".format(fe[0], sense_fes)) # fe[0] = FE name
                i += 2

            ctx_f, ctx_w = populate_contexts(f, "LUs")
            for lu in ctx_f:
                sense_lus = bag_of_words(lu, ctx_w) # # best WN sense for each LUs of each FE
                out.write("Frame lexical unit, {0}, Wordnet Synset, {1}\n".format(lu, sense_lus))

        print("Done. Starting evaluation.")

    # Comparing of the results obtained with the BagOfWords mapping to the gold annotated WN synsets
    # (for each frame element considered)
    evaluate() ###

    
    #### <<<Gold human annotation>>> ###
    #print(get_main_clause("Bond_maturation"))
    wnac = WordNet()
    x = wnac.get_disambiguation_context(get_main_clause("Commutative_process"))
    #print(x)
    
