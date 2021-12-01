#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (Madhav Jariwala / makejari)
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

def calculate_emission_probabilities(train_letters,test_letters):
    emission_probabilities = {}
    emission_count = {}
    for letter in range(len(test_letters)):
        emission_probabilities[letter] = {}
        emission_count[letter] = {}
        for train in train_letters:
            number_matching_pixels_pixels = len([(train_letters[train][x][y],test_letters[letter][x][y]) for x in range(CHARACTER_HEIGHT) \
                            for y in range(CHARACTER_WIDTH) if (train_letters[train][x][y]==test_letters[letter][x][y] and test_letters[letter][x][y]=='*')])
            number_of_empty_pixels = len([(train_letters[train][x][y],test_letters[letter][x][y]) for x in range(CHARACTER_HEIGHT) \
                            for y in range(CHARACTER_WIDTH) if (train_letters[train][x][y]==test_letters[letter][x][y] and test_letters[letter][x][y]==' ')])
            emission_cost = (number_matching_pixels_pixels) * 0.8 + (number_of_empty_pixels) * 0.2
            emission_count[letter][train] = emission_cost
    for letter in range(len(test_letters)):
        for train in train_letters:
            # emission_probabilities[letter][train] = (emission_count[letter][train] + 1) / (sum(emission_count[letter].values())+ len(train_letters) * 2) #referenced from https://www.analyticsvidhya.com/blog/2021/04/improve-naive-bayes-text-classifier-using-laplace-smoothing/
            emission_probabilities[letter][train] = (emission_count[letter][train] + 1) / (sum(emission_count[letter].values())+ 2)
    return emission_probabilities


def calculate_initial_probabilities(data,total_letter_list):
    initial_count = {}
    initial_probabilities = {}
    for letter in total_letter_list:
        initial_count[letter] = 0
    for word in range(len(data)):
        if data[word][0] in total_letter_list:
            initial_count[data[word][0]] += 1
    for dict_letter in initial_count:
        initial_probabilities[dict_letter] = (initial_count[dict_letter] + 1) / (sum(initial_count.values()) + 2)
    return initial_probabilities

def calculate_transition_probabilities(data,total_letter_list):
    added_string = ' '.join([word for word in data])
    # print(added_string)
    # print(len(added_string))
    transition_count = np.zeros((len(total_letter_list),len(total_letter_list)))
    
    for letter in range(len(added_string)-1):
        if added_string[letter] in total_letter_list and added_string[letter+1] in total_letter_list:
            one = total_letter_list.index(added_string[letter])
            two = total_letter_list.index(added_string[letter+1])
            transition_count[one,two]+=1
    rows_sum = np.sum(transition_count,axis=1)

    for i in range(len(total_letter_list)):
        for j in range(len(total_letter_list)):
            transition_count[i,j] = (transition_count[i,j] + 1) / (rows_sum[i]+2)
    return transition_count

    
def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)

    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def read_data(fname):
    exemplars = []
    file = open(fname, 'r')
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += data[0::2]
    return exemplars

def calculate_using_bayes(test_letters,train_letters):    
    emission_probabilities = calculate_emission_probabilities(train_letters,test_letters)
    bayes_string = ''
    for letter in emission_probabilities:
        max_key = max(emission_probabilities[letter], key=emission_probabilities[letter].get)
        bayes_string += max_key
    return bayes_string
    

def calculated_using_viterbi(data,train_word_list,test_letters,train_letters,lines):
    emission_probabilities = calculate_emission_probabilities(train_letters,test_letters)
    initial_probabilities = calculate_initial_probabilities(lines,train_word_list)
    transition_probabilities = calculate_transition_probabilities(data, train_word_list)
    
    # print(initial_probabilities)
    # temp_word_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    
    
    storing_probabilities = np.ones((len(train_word_list),len(test_letters)))
    storing_letter = np.ones((len(train_word_list),len(test_letters)))
    for letter in range(len(test_letters)):    
        for train in range(len(train_word_list)):
            if letter==0:
                storing_probabilities[train][letter] =  np.log(emission_probabilities[letter][train_word_list[train]]) + np.log(initial_probabilities[train_word_list[train]]) * 0.01
                storing_letter[train][letter] = train
            
            else:
                storing_probabilities[train][letter] = np.max([storing_probabilities[t][letter-1]
                        +np.log(transition_probabilities[t,train])*0.01
                        +np.log(emission_probabilities[letter][train_word_list[train]]) for t in range(len(train_word_list))])
                storing_letter[train][letter] = np.argmax([storing_probabilities[t][letter-1]
                        +np.log(transition_probabilities[t,train])*0.01
                        +np.log(emission_probabilities[letter][train_word_list[train]]) for t in range(len(train_word_list))])

    best_pointer =np.argmax([storing_probabilities[t][len(test_letters)-1] for t in range(len(train_word_list))])
    backtrack = []
    temp_best_pointer = best_pointer
    for back_pos in range(len(test_letters),0,-1):
        temp = train_word_list[temp_best_pointer]
        backtrack.append(temp)
        temp_best_pointer = int(storing_letter[temp_best_pointer][back_pos-1])   
    
        
    return ''.join(backtrack[::-1])


#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
data = read_data(train_txt_fname)

f = open(train_txt_fname,'r')
lines = f.readlines()


bayes = calculate_using_bayes(test_letters,train_letters)


viterbi = calculated_using_viterbi(data,TRAIN_LETTERS,test_letters,train_letters,lines)


# The final two lines of your output should look something like this:
print("Simple: " + bayes)
print("   HMM: " + viterbi) 


