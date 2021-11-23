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

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

# def calculate_emission_probabilities(train_letters,test_letters):
#     emission_probabilities = {}
#     emission_count = {}
#     for letter in range(len(test_letters)):
#         emission_probabilities[letter] = {}
#         emission_count[letter] = {}
#         for train in train_letters:
#             number_matching_pixels = len([(train_letters[train][x][y],test_letters[letter][x][y]) for x in range(CHARACTER_HEIGHT) \
#                             for y in range(CHARACTER_WIDTH) if (train_letters[train][x][y]==test_letters[letter][x][y] and test_letters[letter][x][y]=='*')])
#             if len([x for x in range(CHARACTER_HEIGHT) for y in range(CHARACTER_WIDTH) if test_letters[letter][x][y]=='*']) < 20:    
#                 if train==' ' and number_matching_pixels < 5:
#                     emission_count[letter][train] = 5
#                 else:
#                     emission_count[letter][train] = number_matching_pixels
#             else:   
#                 emission_count[letter][train] = number_matching_pixels
#     for letter in range(len(test_letters)):
#         for train in train_letters:
#             prob = round(emission_count[letter][train] / sum(emission_count[letter].values()),6)
#             if prob==0:
#                 emission_probabilities[letter][train] = round(1/99999999,6)
#             else:
#                 emission_probabilities[letter][train] = prob
#     return emission_probabilities

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
            emission_cost = (number_matching_pixels_pixels) * 0.80 + (number_of_empty_pixels) * 0.20
            emission_count[letter][train] = emission_cost
    for letter in range(len(test_letters)):
        for train in train_letters:
            prob = round(emission_count[letter][train] / sum(emission_count[letter].values()),6)
            if prob==0:
                emission_probabilities[letter][train] = round(1/99999999,6)
            else:
                emission_probabilities[letter][train] = prob
    return emission_probabilities

def calculate_initial_probabilities(data,total_letter_list):
    initial_count = {}
    initial_probabilities = {}
    for letter in total_letter_list:
        initial_count[letter] = 0
    for word in data:
        if word[0] in total_letter_list:
            initial_count[word[0]] += 1
    for dict_letter in initial_count:
        prob = round(initial_count[dict_letter] / sum(initial_count.values()),6)
        if prob==0:
            initial_probabilities[dict_letter] = round(1/99999999,6)
        else:
            initial_probabilities[dict_letter] = prob
    return initial_probabilities


def calculate_transition_probabilities(data,total_letter_list):
    transition_count = {}
    transition_probabilities = {}
    for letter in total_letter_list:
        transition_count[letter] = {}
        transition_probabilities[letter] = {}
        for next_letter in total_letter_list:
            transition_count[letter][next_letter] = 0
    for word in data:
        for word_index in range(1,len(word)):
            if (word[word_index] in total_letter_list) and (word[word_index-1] in total_letter_list):
                transition_count[word[word_index-1]][word[word_index]] += 1
    for letter in total_letter_list:
        for next_letter in total_letter_list:
            if transition_count[letter] == 0:
                prob = round(transition_count[letter][next_letter] / sum(transition_count[letter].values()),6)
            else:
                prob = round(transition_count[letter][next_letter] / 1,6)
            if prob == 0:
                transition_probabilities[letter][next_letter] = round(1/99999999,6)
            else:
                transition_probabilities[letter][next_letter] = prob
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
    
def calculated_using_viterbi(data,train_word_list,test_letters,train_letters):
    initial_probabilities = calculate_initial_probabilities(data, TRAIN_LETTERS)
    transition_probabilities = calculate_transition_probabilities(data, TRAIN_LETTERS)

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
data = read_data(train_txt_fname)

bayes = calculate_using_bayes(test_letters,train_letters)


viterbi = calculated_using_viterbi(data,TRAIN_LETTERS,test_letters,train_letters)
# for i in train_letters:
    # print("\n".join([ r for r in train_letters[i]]))

# for letter in range(len(test_letters)):
#     print(letter,len([x for x in range(CHARACTER_HEIGHT) for y in range(CHARACTER_WIDTH) if test_letters[letter][x][y]=='*']))





# len("Train letter Length:",train_letters)
# len("Test letter Length:",test_letters)
## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print("\n".join([ r for r in test_letters[2] ]))



# The final two lines of your output should look something like this:
print("Simple: " + bayes)
print("   HMM: " + "Sample simple result") 


