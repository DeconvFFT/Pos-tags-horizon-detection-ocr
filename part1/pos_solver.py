###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    pos_tag_list = ['det', 'adj', 'prt', '.', 'verb', 'num', 'pron', 'x', 'conj', 'adp', 'adv', 'noun']
    initial_probabilites = {}
    transition_probabilities = {}
    emission_probabilities = {}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "HMM":
            return -999
        elif model == "Complex":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def calculate_count_initial_probabilites(self,tags,data):
        initial_probabilites = {}
        initial_count = {}
        for j in range(len(tags)):
            initial_count[tags[j]] = 0
            initial_probabilites[tags[j]] = 1/999999999999999
        for i in range(len(data)):
            initial_count[tags[tags.index(data[i][1][0])]] += 1
        for key in initial_count:
            initial_probabilites[key] = round(initial_count[key] / sum(initial_count.values()),6)
        return initial_probabilites

    def calculate_count_emission_probabilities(self,tags,data):
        emission_probabilities = {}
        emission_count = {}
        emission_word_count = {}
        words_list = set(x for li in data for s in li for x in s)
        for i in words_list:
            emission_word_count[i] = 0
            emission_count[i] = {}
            emission_probabilities[i] = {}
            for k in tags:
                emission_count[i][k] = 0
                emission_probabilities[i][k] = 1/999999999999999
        for i in range(len(data)):
            for j in range(len(data[i][0])):
                emission_word_count[data[i][0][j]] += 1
                emission_count[data[i][0][j]][data[i][1][j]] += 1
        for i in range(len(data)):
            for j in range(len(data[i][0])):
                emission_probabilities[data[i][0][j]][data[i][1][j]] = emission_count[data[i][0][j]][data[i][1][j]]/emission_word_count[data[i][0][j]]
        
        return emission_probabilities

    def transition_count_probabilities(self,tags,data):
        transition_probabiity = {}
        transition_count = {}
        for i in tags:
            transition_count[i] = {}
            transition_probabiity[i] = {}
            for j in tags:
                transition_count[i][j] = 0
                transition_probabiity[i][j] = 1/999999999999999
        for i in range(len(data)):
            for j in range(1,len(data[i][1])):
                transition_count[data[i][1][j-1]][data[i][1][j]] += 1
        for key in tags:
            for pos in tags:
                transition_probabiity[key][pos] = transition_count[key][pos]/sum(transition_count[key].values())
        return transition_probabiity

    def train(self, data):
        #code added
        
        self.initial_probabilites = self.calculate_count_initial_probabilites(self.pos_tag_list,data)
        self.emission_probabilities = self.calculate_count_emission_probabilities(self.pos_tag_list,data)
        self.transition_probabilities = self.transition_count_probabilities(self.pos_tag_list,data)

        print("Initial_prob:",self.initial_probabilites)
        print("Emission_prob:",self.emission_probabilities)
        print("Transition_prob:",self.transition_probabilities)
        

        #code added ended

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        return [ "noun" ] * len(sentence)

    def complex_mcmc(self, sentence):
        return [ "noun" ] * len(sentence)



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

