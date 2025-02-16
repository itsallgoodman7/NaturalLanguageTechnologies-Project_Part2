from WordNet import WordNet
from Metrics import Metrics
from PS_Index import *
import time
import os

output = os.path.dirname(__file__) + '/output/'
def parse_word_sim_353(path):
    result = []
    with open(path, 'r') as file:
        for line in file.readlines()[1:]:
            temp = line.split(",")
            score = temp[2].replace('\n', '')
            result.append((temp[0], temp[1], float(score)))

    return result


def conceptual_similarity():

    ws353 = parse_word_sim_353('WordSim353.csv')
    print("[1] - WordSim353.csv parsed.")

    wn = WordNet()

    similarities = []  # lista di liste (una per ogni couple_terms) di similarità, una lista per ogni metric
    metric_obj = Metrics(wn)
    time_start = time.time()

    # A list of 2 tuples: the first containing the reference to the metrics
    # implementation, and the second containing his name (a tuple of 3 strings).
    metrics = list(zip(*metric_obj.get_all()))
    #print(metrics) # [(<bound method Metrics.wu_palmer_metric of <Metrics.Metrics object at 0x7fb029722a90>>, <bound method Metrics.shortest_path_metric of <Metrics.Metrics object at 0x7fb029722a90>>, <bound method Metrics.leakcock_chodorow_metric of <Metrics.Metrics object at 0x7fb029722a90>>), ('Wu & Palmer', 'Shortest Path', 'Leakcock & Chodorow')]

    to_remove = []
    count_total_senses = 0  # to count the senses total

    # Looping over the list of all the three metrics.
    for metric in metrics[0]:
        sim_metric = []  # similarity list for this metric

        j = 0
        for couple_terms in ws353:
            synset1 = wn.get_synsets(couple_terms[0])
            synset2 = wn.get_synsets(couple_terms[1])
            # senses = [synset1, synset2]

            maxs = []  # list of senses similarity between the two senses considered at each time
            for s1 in synset1:
                for s2 in synset2:
                    count_total_senses += 1
                    maxs.append(metric(s1, s2)) # tutte le misure di similarità considerate per ogni senso di ogni couple_terms
            if len(maxs) == 0:  # word without senses (ex.: proper nouns)
                maxs = [-1]
                to_remove.append(j)
            sim_metric.append(max(maxs)) # massime similarità per ogni couple_terms
            j += 1
        similarities.append(sim_metric) # lista di tutte le massime misure di similarità per ogni couple_terms (i sensi massimamente simili), per ogni metrica considerata

    time_end = time.time()
    print("[1] - Total senses similarity: {}".format(count_total_senses))
    print("[1] - Time elapsed: {0:0.2f} seconds".format(time_end - time_start))

    for j in range(len(ws353)):
        if j in to_remove:
            del ws353[j]
            for s in range(len(similarities)):
                del similarities[s][j]

    golden = [row[2] for row in ws353]  # the list of golden annotations (scores column)

    pearson_list = []
    spearman_list = []

    for i in range(len(metrics[1])):
        yy = similarities[i] # similarity list associata alla singola misura ogni volta
        pearson_list.append(pearson_index(golden, yy))
        spearman_list.append(spearman_index(golden, yy))

    with open(output + 'task1_results.csv', "w") as out:
        out.write("word1, word2, {}, {}, {}, gold\n"
                  .format(metrics[1][0], metrics[1][1], metrics[1][2]))
        for j in range(len(ws353)):
            out.write("{0}, {1}, {2:.2f}, {3:.2f}, {4:.2f}, {5}\n"
                      .format(ws353[j][0], ws353[j][1], similarities[0][j],
                              similarities[1][j], similarities[2][j], ws353[j][2], )
                      )

    with open(output + 'task1_indices.csv', "w") as out:
        out.write(" , Pearson, Spearman\n")
        for j in range(len(pearson_list)):
            out.write("{}, {}, {}\n".format(metrics[1][j], str(pearson_list[j]), spearman_list[j]))


conceptual_similarity()