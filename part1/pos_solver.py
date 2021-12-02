###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
from typing import final
import numpy as np
from numpy.core.records import get_remaining_size
from numpy.lib.function_base import sinc
import operator
from collections import Counter
from random import random
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    pos_tag_list = ['det', 'adj', 'prt', '.', 'verb', 'num', 'pron', 'x', 'conj', 'adp', 'adv', 'noun']
    initial_probabilites = {}
    transition_probabilities = {}
    emission_probabilities = {}
    joint_probs = {}
    joint_counts = {}
    
    iterations = 500
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            bayes_string = []
            posterior_prob = 1
            for letter in sentence:
                if letter in self.emission_probabilities:
                    max_key = max(self.emission_probabilities[letter], key=self.emission_probabilities[letter].get)
                    bayes_string.append(max_key)
                    posterior_prob = posterior_prob + self.emission_probabilities[letter][max_key]
                else:
                    bayes_string.append("noun")
                    posterior_prob = posterior_prob + 1/12
            
            return posterior_prob
            #return -999
            
            
        elif model == "HMM":
            posterior_prob = 1
            storing_probabilities = np.ones((len(self.pos_tag_list),len(sentence)))
            storing_letter = np.ones((len(self.pos_tag_list),len(sentence)))
            
            for letter in range(len(sentence)):
                if sentence[letter] in self.emission_probabilities:
                    for train in range(len(self.pos_tag_list)):
                        if letter==0:
                            storing_probabilities[train][letter] =  np.log(self.emission_probabilities[sentence[letter]][self.pos_tag_list[train]]) 
                            + (np.log(self.initial_probabilites[self.pos_tag_list[train]]))
                            storing_letter[train][letter] = train
                        
                        else:
                            storing_probabilities[train][letter] = np.max([storing_probabilities[t][letter-1]
                                    +np.log(self.transition_probabilities[self.pos_tag_list[t]][self.pos_tag_list[train]])
                                    +np.log(self.emission_probabilities[sentence[letter]][self.pos_tag_list[train]]) for t in range(len(self.pos_tag_list))])
                            storing_letter[train][letter] = np.argmax([storing_probabilities[t][letter-1]
                                    +np.log(self.transition_probabilities[self.pos_tag_list[t]][self.pos_tag_list[train]])
                                    +np.log(self.emission_probabilities[sentence[letter]][self.pos_tag_list[train]]) for t in range(len(self.pos_tag_list))])
                            
                #if new word:
                else:
                    for train in range(len(self.pos_tag_list)):
                        if letter==0:
                            storing_probabilities[train][letter] =  1/len(self.pos_tag_list)
                            + (np.log(self.initial_probabilites[self.pos_tag_list[train]]))
                            storing_letter[train][letter] = train
                        
                        
                        else:
                            storing_probabilities[train][letter] = np.max([storing_probabilities[t][letter-1]
                                    +np.log(self.transition_probabilities[self.pos_tag_list[t]][self.pos_tag_list[train]])
                                    +np.log(1/len(self.pos_tag_list)) for t in range(len(self.pos_tag_list))])
                            storing_letter[train][letter] = np.argmax([storing_probabilities[t][letter-1]
                                    +np.log(self.transition_probabilities[self.pos_tag_list[t]][self.pos_tag_list[train]])
                                    +np.log(1/len(self.pos_tag_list)  
                                            ) for t in range(len(self.pos_tag_list))])

            best_pointer =np.argmax([storing_probabilities[t][len(sentence)-1] for t in range(len(self.pos_tag_list))])
            
            
                        
            backtrack = []
            temp_best_pointer = best_pointer
            # for back_pos in range(len(sentence),0,-1):
                
            #     #print(posterior_prob)
            #     posterior_prob = posterior_prob + np.log( storing_probabilities[temp_best_pointer][back_pos-1])
                                
            #     temp = self.pos_tag_list[temp_best_pointer]
            #     backtrack.append(temp)
            #     temp_best_pointer = int(storing_letter[temp_best_pointer][back_pos-1])   
            
            #print("Last value:::", np.exp(storing_probabilities[best_pointer]))
            
            return -storing_probabilities[best_pointer][-1]
            #return -999
        
        
        elif model == "Complex":
            return self.get_complex_probs(list(sentence), list(label))
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
            #initial_probabilites[key] = (initial_count[key] + 1) / (sum(initial_count.values()) + 2)
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
                
                #emission_probabilities[data[i][0][j]][data[i][1][j]] = (emission_count[data[i][0][j]][data[i][1][j]] +1) /(emission_word_count[data[i][0][j]] + 2)
        
        
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
        #############
        #rows_sum = np.sum(transition_count,axis=1)
        
        for key in tags:
            for pos in tags:
        
        
                transition_probabiity[key][pos] = transition_count[key][pos]/sum(transition_count[key].values())
                #transition_probabiity[key,pos] = (transition_probabiity[key,pos] + 1) / (rows_sum[key]+2)
        
        
        return transition_probabiity
    
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
        #############
        #rows_sum = np.sum(transition_count,axis=1)
        
        for key in tags:
            for pos in tags:
        
        
                transition_probabiity[key][pos] = transition_count[key][pos]/sum(transition_count[key].values())
                #transition_probabiity[key,pos] = (transition_probabiity[key,pos] + 1) / (rows_sum[key]+2)
        
        
        return transition_probabiity

    def create_joint_counts(self, tag1, tag2, tag3):
        if tag1 in self.joint_counts:
            if tag2 in self.joint_counts[tag1]:
                if tag3 in self.joint_counts[tag1][tag3]:
                    self.joint_counts[tag1][tag2][tag3] = self.joint_counts[tag1][tag2][tag3] + 1
                else:
                    self.joint_counts[tag1][tag2][tag3]= 1
            else:
                self.joint_counts[tag1][tag2] = {tag3 : 1}
        else:
            self.joint_counts[tag1]= {tag2:{tag3 : 1}}
            
    def get_joint_probs(self, tag1, tag2, tag3):
        if tag1 in self.joint_counts and tag2 in self.joint_counts[tag1] and tag3 in self.joint_counts[tag1][tag2]:
            prob = self.join_counts[tag1][tag2][tag3] / np.sum(self.joint_counts[tag1][tag2].values())
            self.join_probs[tag1] = {tag2 : {tag3 : prob}}
            return prob
        return 0.0000000001
    
    def get_complex_probs(self, words, sample):
        s1 = sample[0]
        cost_s1 = np.log(self.initial_probabilites[s1])
        w_emission = 0
        pos_trans = 0
        pos_trans_2 = 0

        for i in range(len(sample)):
            if words[i] in self.emission_probabilities:
                w_emission += np.log(self.emission_probabilities[words[i]][sample[i]])
            else:
                w_emission+=np.log(1/12)
            if i != 0:
                pos_trans += np.log(self.transition_probabilities[sample[i - 1]][sample[i]])
            if i != 0 and i != 1:
                pos_trans_2 += np.log(self.get_joint_probs(sample[i - 2], sample[i - 1], sample[i]))
        return cost_s1 + w_emission+ pos_trans+ pos_trans_2
    
    
    def get_random(self,prob):
        return 1 if random() <= prob else 0
    
    def generate_samples(self, words, sample,tagset):
        for index in range(len(words)):
            probs = [0] * len(tagset)
            for j in range(len(tagset)):
                sample[index] = tagset[j]
                probs[j] = np.exp(self.get_complex_probs(words, sample))
                
            probs/=sum(probs)
            rand = random()
            p = 0
            for i in range(len(probs)):
                p += probs[i]
                if rand < p:
                    sample[index] = tagset[i]
                    break
        return sample
    
    def mcmc(self, words):
        samples = []
        sample = self.simplified(words)
        for i in range(self.iterations):
            sample = self.generate_samples(words, sample, self.pos_tag_list)
            samples.append(sample)
        samples_occurence_list = []
        for i in range(len(words)):
            sample_occurence = {}
            if sample[i] in sample_occurence:
                sample_occurence[sample[i]] +=1
            else:
                sample_occurence[sample[i]] = 1
            samples_occurence_list.append(sample_occurence)
        tags = [max(samples_occurence_list[i], key = samples_occurence_list[i].get) for i in range(len(words))]
        return tags
    
    def train(self, data):
        #code added
        
        self.initial_probabilites = self.calculate_count_initial_probabilites(self.pos_tag_list,data)
        self.emission_probabilities = self.calculate_count_emission_probabilities(self.pos_tag_list,data)
        self.transition_probabilities = self.transition_count_probabilities(self.pos_tag_list,data)

        self.data = data
        # print("Initial_prob:",self.initial_probabilites)
        # print("Transition_prob:",self.transition_probabilities)
        

        #code added ended

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        bayes_string = []
        for letter in sentence:
            if letter in self.emission_probabilities:
                max_key = max(self.emission_probabilities[letter], key=self.emission_probabilities[letter].get)
                bayes_string.append(max_key)
            else:
                bayes_string.append("noun")
                
        return bayes_string
        

    def hmm_viterbi(self, sentence):
                
        storing_probabilities = np.ones((len(self.pos_tag_list),len(sentence)))
        storing_letter = np.ones((len(self.pos_tag_list),len(sentence)))
        
        for letter in range(len(sentence)):
            if sentence[letter] in self.emission_probabilities:
                for train in range(len(self.pos_tag_list)):
                    if letter==0:
                        storing_probabilities[train][letter] =  np.log(self.emission_probabilities[sentence[letter]][self.pos_tag_list[train]]) 
                        + (np.log(self.initial_probabilites[self.pos_tag_list[train]]))
                        storing_letter[train][letter] = train
                    
                    else:
                        storing_probabilities[train][letter] = np.max([storing_probabilities[t][letter-1]
                                +np.log(self.transition_probabilities[self.pos_tag_list[t]][self.pos_tag_list[train]])
                                +np.log(self.emission_probabilities[sentence[letter]][self.pos_tag_list[train]]) for t in range(len(self.pos_tag_list))])
                        storing_letter[train][letter] = np.argmax([storing_probabilities[t][letter-1]
                                +np.log(self.transition_probabilities[self.pos_tag_list[t]][self.pos_tag_list[train]])
                                +np.log(self.emission_probabilities[sentence[letter]][self.pos_tag_list[train]]) for t in range(len(self.pos_tag_list))])
                        
            #if new word:
            else:
                for train in range(len(self.pos_tag_list)):
                    if letter==0:
                        storing_probabilities[train][letter] =  1/len(self.pos_tag_list)
                        + (np.log(self.initial_probabilites[self.pos_tag_list[train]]))
                        storing_letter[train][letter] = train
                    
                    
                    else:
                        storing_probabilities[train][letter] = np.max([storing_probabilities[t][letter-1]
                                +np.log(self.transition_probabilities[self.pos_tag_list[t]][self.pos_tag_list[train]])
                                +np.log(1/len(self.pos_tag_list)) for t in range(len(self.pos_tag_list))])
                        storing_letter[train][letter] = np.argmax([storing_probabilities[t][letter-1]
                                +np.log(self.transition_probabilities[self.pos_tag_list[t]][self.pos_tag_list[train]])
                                +np.log(1/len(self.pos_tag_list)
                                        
                                        ) for t in range(len(self.pos_tag_list))])

        best_pointer =np.argmax([storing_probabilities[t][len(sentence)-1] for t in range(len(self.pos_tag_list))])
        backtrack = []
        temp_best_pointer = best_pointer
        for back_pos in range(len(sentence),0,-1):
            temp = self.pos_tag_list[temp_best_pointer]
            backtrack.append(temp)
            temp_best_pointer = int(storing_letter[temp_best_pointer][back_pos-1])   
        
        
        return backtrack[::-1]
            
        
        



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
            return self.mcmc(list(sentence))
        else:
            print("Unknown algo!")

