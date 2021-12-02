# Part 1: POS Tagging

## A description of how you formulated the search problem
This problem has three subparts in it: First part where we are asked to implement POS Tagger using bayes algorithm, the second part where we are asked to implement POS Tagger using Viterbi Algorithm and the third part where we are asked to implement using Gibbs Sampling. For the bayes, we used only emission probability to calculate the maximum values from all possible outcomes, comparing all 12 tags. In the second part we implemented Viterbi Algorithm and we took the max value of Emission, Transition and Initial Probabilities and the results were slighly better as transition probability was allowing the algorithm to pick the most common POS tags of given words.

### Emission Probability
Emission probability was calculated by checking each word and its occurence and the associated POS tag. One dictionary was created to register number of times the certain POS tag was associated with the given word. Then, the probability was calculated by diving the count with the number of occurences of that word. 

### Transition Probability
Transition Probability was calcualted by iterating over the whole sentence and starting from the second word as it first word won've have any word before it and it would be used to calculate initial probability. The second word we would consider as #not complete from here

### Initial Probability
Initial count was calculated by taking the count of the first letter from each sentence from "bc.train" file. From this, the initial probability was calculated by Laplace smoothing.

## A brief description of how your program works
For the bayes algorithm, we simply calculated the emission probability and returned the max values for each letter. For the Viterbi Algorithm, we calculated Emission, Initial and Transition probability and we took log of them and took sum because `log(a*b) = log(a) + log(b)`. The reason why we took log was because the values were underflowing as we iterated over few letter/observed states. Then we created two matrices: one to store the probabiliy and second to store the path which would help us to backtrack and calculate the maximum cost path. In probability matrix, we stored the `max` values and in the backtrack matrix, we stored index of the path that we chose for the best path using `argmax`. In Viterbi, for the first letter, we only used Emission and Initial Probabiliy and stored the values inside the probability matrix and for the backtrack matrix, we just stored the hidden state/letter as it is the first letter of the sentence. For the rest of the letters, we used all three probabilities to calculate the maximum value. Once iteration was over, we backtracked from the last letter/observed state to the first letter to find the optimal math. 

## Problems faced, assumptions, simplifications and design decisions
The major decision we had to make was optimizing the viterbi algorithm as results we were getting were not good especially when there was a noisy image. Also, the training set we used was `bc.train` and the transition probability was sometimes messing up the result and giving dump values. So we had to make a decision to multiply the value that we get from the transition probability with 0.01 as that would make it significance less and it was surprisingly giving me good results and improving the result that we got from bayes. The same thing was happening with initial probabilities as well as some letter were correctly selected in bayes but in viterbi, it slighly relied on initial probabilities more. Especially for letter such as "T","I",etc. So we had to multiply it with the 0.01 value to make its value less significant. We believe it is because we are using laplace smoothing, it would bring values closer to each other. 

