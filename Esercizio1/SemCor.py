import re
import sys
import xml.etree.ElementTree as ET
from lesk import *
from lxml import etree as Exml
from tqdm import tqdm
import random
import os


input = os.path.dirname(__file__) + '/semcor3.0/brown1/tagfiles/br-a01'
output = os.path.dirname(__file__) + '/output/'
average_accuracy = 0



def parse_xml(path):
    """
    It parses the SemCor corpus, which has been annotated by hand on WordNet
    Sysnsets.

    In order:
    1) Load XML file (root, paragraphs and words)
    2) Took all the tags "s" (substantive)
    3) Extract the sentence (sent)
    4) Select the words to disambiguate (select only the needed ones, which have multiple senses)
    5) Extract Golden annotated sense from WSN (word.attrib['wnsn'])

    :param path: the path to the XML file (Brown Corpus)
    :return: [(sentence, [(word, gold)])]
    """

    with open(path, 'r') as fileXML:
        data = fileXML.read()

        # fixing XML's bad formatting
        data = data.replace('\n', '')
        replacer = re.compile("=([\w|:|\-|$|(|)|']*)")
        data = replacer.sub(r'="\1"', data) # \1 is the replacement to use in case of a match, 
                                            #so that a repeated word will be replaced by a single word.
        result = []
        try:
            root = Exml.XML(data)
            paragraphs = root.findall("./context/p")
            sentences = []
            for p in paragraphs:
                sentences.extend(p.findall("./s"))
            for sentence in sentences:
                words = sentence.findall('wf')
                sent = ""
                tuple_list = [] # list of all the tuple ((word, gold_sense/wnsn in SemCore))
                for word in words:
                    w = word.text
                    pos = word.attrib['pos'] #search POS tagging attribute of word
                    sent = sent + w + ' ' # building a string containing all the words parsed in the sentences (for all paragraph)
                    # if word's POS tag attribute is "NN" = substantive and word have multiple senses and WordNet synsets attribute for word exists 
                    if pos == 'NN' and '_' not in w \
                            and len(wn.synsets(w)) > 1 \
                            and 'wnsn' in word.attrib: # wnsn = WordNet synset annotated in the SemCor corpus as the "gold" sense
                        sense = word.attrib['wnsn']
                        t = (w, sense) # tuple (word, gold_sense)
                        tuple_list.append(t) # building the list of all the tuple considered
                result.append((sent, tuple_list)) # adding the tuple relative to the paragraph p just parsed
        except Exception as e:
            raise NameError('xml: ' + str(e))
    return result # a list of tuple [ ("string_all_parsed_words_in_sentences", [(word1,wnsn), (word2, wnsn), (...),...]), (...), ...] ) ]
                  # with one tuple for each paragraph parsed in SemCor corpus
    

def word_sense_disambiguation():
    """
    Word Sense Disambiguation: Extracts sentences from the SemCor corpus
    (corpus annotated with WN synset) and disambiguates at least one ambiguous noun
    (words with POS_tag 's' and multiple WN synsets) per sentence.
    It also calculates the accuracy based on the senses noted in SemCor.
    Writes the output into a xml file.

    """

    ##list_xml = parse_xml(options["input"])
    list_xml = parse_xml(input)

    result = []
    count_word = 0
    count_exact = 0

    # showing progress bar
    #progress_bar = tqdm(desc="Percentage", total=50, file=sys.stdout)

    
    randomizer = random.sample(range(0, len(list_xml)-1),50) # extract 50 random sentences from the SemCor corpus parsed (randomize tuple choice in result)
    ##while i in range(len(list_xml)) and len(result) < 50:
    for i in randomizer:
        if len(result) < 50:
            dict_list = []
            sentence = list_xml[i][0] # i^ paragraph, first elem of the result tuple (sent) = " sent + w + ' ' "
            # randomize the choosing of t (the substantive to disambiguate among the t words)
            words = list_xml[i][1]  # i^ paragraph, second elem of the result tuple (tuple_list) = [(word1,wnsn), (word2,wnsn), (...)]
            if len(words) != 0:
                randomizer2 = random.randrange(0, (len(words)))
                ###print(words)                        
                #randomizer2 = random.choice(range(words))
                t = words[randomizer2]
                #print(words[randomizer2])
                ###print(t)
                sense = lesk(t[0], sentence)  # running lesk's algorithm on word t[0] and its context sentence
                                                                # -> returns its best sense in context
                value = str(get_sense_index(t[0], sense)) # returning index (among the WN synsets list) of the word sense just found
                golden = t[1] # second elem of the tuple t (words) = wnsn (gold_score)
                count_word += 1
                if golden == value:
                    count_exact += 1
                dict_list.append({'word': t[0], 'gold': golden, 'value': value}) # {1: word, 2: wnsn/golden_sense, 3: lesk_sense}

                if len(dict_list) > 0:
                    result.append((sentence, dict_list)) # [ (word_context, word_evaluation/dict_list), ... ]
                    #progress_bar.update(1)
            else: # empty 
                i = i + 1

    accuracy = count_exact / count_word      # correct / total 

    with open(output + 'task2_output.xml', 'wb') as out:
        out.write('<results accurancy="{0:.2f}">'.format(accuracy).encode())
        for j in range(len(result)):
            xml_s = ET.Element('sentence_wrapper')
            xml_s.set('sentence_number', str(j + 1))
            xml_sentence = ET.SubElement(xml_s, 'sentence')
            xml_sentence.text = result[j][0] # word_context
            for tword in result[j][1]: # word_evaluation
                xml_word = ET.SubElement(xml_sentence, 'word')
                xml_word.text = tword['word']
                xml_word.set('golden', tword['gold'])
                xml_word.set('sense', tword['value'])

            tree = ET.ElementTree(xml_s)
            tree.write(out)

        out.write(b'</results>')
        print(f"Accuracy:{accuracy}")

        return accuracy


print("\nTask 2: Word Sense Disambiguation")
for i in range (0,10):
    print(f"[ {i} ] - Running Lesks's algorithm...")
    average_accuracy+=word_sense_disambiguation()
print(f"Average accuracy:{average_accuracy/10}")