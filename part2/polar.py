#!/usr/local/bin/python3
#
# Authors: [Saumya Hetalbhai Mehta mehtasau]
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
from scipy.stats import norm
import numpy as np

# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# Calculate emission probability
# This function takes image and edge strength as input 
# and returns emission probability for that image
# @param: image: Image array
# @param: edge_strength: Gives us a derviative of image. Higher the value
# in edge strength, more likely a point is to be a part of a boundary
# We take the ratio of edge strength to image intensity to calculate emission probabilities
def get_emission(image, edge_strength):
    nrows = edge_strength.shape[0]
    ncols = edge_strength.shape[1]
    emission = np.zeros((nrows, ncols))
    
    for r in range(nrows):
        for c in range(ncols):
            emission[r][c] = edge_strength[r][c]/image[r][c]
    emission_probs = emission/np.sum(emission, axis = 0)
    return emission_probs



# generate transition probabilities for a row to all rows in next column
# This function takes row of column 1 and number of rows as input and returns the inverse 
# exponential distance probability
# @param: row1: Row of current column
# @param: row2: Row of next column
"""
        Some other distance distributions tried:
        1.) len - dist)/len
        2.) 1/(dist+1)
"""
def get_transition_prob(row1, row2):
     
    dist = abs(row1-row2)
    if dist <=6:
        return -np.log(1/((dist+1)))
    else:
        return -np.log((1/(2**(dist+1)))**2) 
        
# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#

## Assumptions for constructing a boundary...
# air-ice boundary >= ice-rock boundary -10 px
# two boundaries span the entire width of image
# each boundary is relatively smooth: boundary's row in one
# column is similar in the next column
# boundary will be generally dark and along strong image edge (sharp change in pixel values)


def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)

# generate air-ice and ice-rock boundary using simple bayes
# This function takes edge strength map as input and  
# returns list of rows containining boundary points for air-ice and 
# a list containing boundary points for ice-rock
# @param: edge_strength: Gives us a derviative of image. Higher the value
# in edge strength, more likely a point is to be a part of a boundary.
# We calculate the air ice boundary by taking maximum value in a colum
# We calculate the ice-rock boundary by taking maximum value 10px below 
# air ice boundary
def get_bayes_rows(edge_strength):
    max_rows_list = []
    next_rows_list = []
    for col in range(edge_strength.shape[1]):
        edge_col = edge_strength[:,col]
        edge_col_tmp = sorted(edge_col)
        max1 = edge_col_tmp[-1]
        idx1 = np.where(edge_col == max1)[0][0]
        max_rows_list.append(idx1)
    median_air_ice_row = int(median(max_rows_list))
    for col in range(edge_strength.shape[1]):
        next_rows_list+=[(np.argmax(edge_strength[median_air_ice_row+10:,col]))]
    next_rows_list = [row + median_air_ice_row+10 for row in next_rows_list]
    return (max_rows_list, next_rows_list)

# generate air-ice and ice-rock boundary using hidden markov model(Viterbi)
# This function takes edge strength map and image array as input and  
# returns list of rows containining boundary points for air-ice and 
# a list containing boundary points for ice-rock
# @param: edge_strength: Gives us a derviative of image. Higher the value
# in edge strength, more likely a point is to be a part of a boundary
# @param: image: Image array
# using viterbi algorithm, we find the air-ice boundary points 
# and then we get ice-rock boundary by performing viterbi for rows 
# 10 px below air-ice row. 
# We consider minimum cost instead of maximum probability to avoid underflow issues

def get_hmm_rows(image,edge_strength):
    edge_strength = edge_strength+1
    ncols = edge_strength.shape[1]
    nrows = edge_strength.shape[0]
    min_state = np.zeros((nrows, ncols))

    V_table = np.ones((nrows, ncols))
    emission_probs = -np.log(get_emission(image, edge_strength))
    V_table[:,0] = -np.log(edge_strength[:,0]/np.sum(edge_strength[:,0]))+ emission_probs[:,0]

    # Calculation for air ice boundary
    for t in range(1, edge_strength.shape[1]):
        for row in range(edge_strength.shape[0]):
            min_cost = np.Infinity
            for r1 in range(edge_strength.shape[0]):

                distance_distr = get_transition_prob(row, r1)
                transition = distance_distr
                val = V_table[r1][t-1]+transition 
                if(val < min_cost):
                    min_cost = val
                    min_state[row][t] = r1
                    V_table[row][t] = min_cost + emission_probs[row][t]

    bkptr = np.argmin(V_table[:,-1])
    ridge_air_ice = np.zeros(ncols)
    for col in range(ncols-1, -1, -1):
        ridge_air_ice[col] = int(bkptr)
        bkptr = min_state[int(bkptr)][col]

    # # Calculation for ice-rock boundary

    V_table_ir = np.ones((nrows, ncols))*np.Infinity
    min_state = np.zeros((nrows, ncols))
    for row in range(int(ridge_air_ice[0])+10, edge_strength.shape[0]):
        V_table_ir[row,0] = -np.log(edge_strength[row][0]/np.sum(edge_strength[:,0])) + emission_probs[row,0]

    for col in range(ncols):
        V_table_ir[int(ridge_air_ice[col]),col] = np.Infinity
    
    for t in range(1, edge_strength.shape[1]):
            for row in range(int(ridge_air_ice[t])+10, edge_strength.shape[0]):
                min_cost = np.Infinity
                for r1 in range(edge_strength.shape[0]):

                    distance_distr = get_transition_prob(row, r1)
                    #transition = distance_distr
                    transition = distance_distr
                    val = V_table_ir[r1][t-1]+transition 
                    if(val < min_cost):
                        min_cost = val
                        min_state[row][t] = r1
                        V_table_ir[row][t] = min_cost + emission_probs[row][t]
                    
    bkptr = np.argmin(V_table_ir[:,-1])
    ridge_ice_rock = np.zeros(ncols)
    for col in range(ncols-1, -1, -1):
        ridge_ice_rock[col] = int(bkptr)
        bkptr = min_state[int(bkptr)][col]
    return (ridge_air_ice,ridge_ice_rock)

# generate air-ice boundary using hidden markov model(Viterbi) and 
# human feedback
# This function takes edge strength map and image array as input and  
# returns list of rows containining boundary points for air-ice and 
# a list containing boundary points for ice-rock
# @param: edge_strength: Gives us a derviative of image. Higher the value
# in edge strength, more likely a point is to be a part of a boundary
# @param: image: Image array
# @param: row_coord: Row number input by human for air ice boundary
# @param: col_coord: Column number input by human for air ice boundary
# using viterbi algorithm, we find the air-ice boundary points.
# We run viterbi in two directions, 1.) from col_coord till all we reach the 
# end of columns. 2.) From col_coord to the start of columns.
# We consider minimum cost instead of maximum probability to avoid underflow issues

def get_feedback_rows_air_ice(edge_strength, image,row_coord, col_coord):
    edge_strength = edge_strength+1
    nrows = edge_strength.shape[0]
    ncols = edge_strength.shape[1]
    
    emission_probs = edge_strength/sum(edge_strength, axis = 0)
    emission_probs = -np.log(emission_probs)
    #emission_probs = -np.log(get_emission(image, edge_strength))

    #create viterbi table
    V_table = np.ones((edge_strength.shape[0], edge_strength.shape[1]))
    V_table[:,col_coord] = np.Infinity
    V_table[row_coord,col_coord] = -np.log(1)
    min_state = np.zeros((nrows, ncols))

    for t in range(col_coord-1, 0,-1):
        for row in range(0, nrows):
            min_cost = np.Infinity
            for r1 in range(edge_strength.shape[0]):

                distance_distr = get_transition_prob(row, r1)
                transition = distance_distr
                val = V_table[r1][t+1]+transition 
                if(val < min_cost):
                    min_cost = val
                    min_state[row][t] = r1
                    V_table[row][t] = min_cost + emission_probs[row][t]
    
    for t in range(col_coord+1, edge_strength.shape[1]):
        for row in range(edge_strength.shape[0]):
            min_cost = np.Infinity
            for r1 in range(edge_strength.shape[0]):

                distance_distr = get_transition_prob(row, r1)
                transition = distance_distr
                val = V_table[r1][t-1]+transition 
                if(val < min_cost):
                    min_cost = val
                    min_state[row][t] = r1
                    V_table[row][t] = min_cost + emission_probs[row][t]
            
    backptr = np.argmin(V_table[:,-1])
    ridge = np.zeros(ncols)
    for col in range(ncols-1, -1, -1):
        ridge[col] = int(backptr)
        backptr = min_state[int(backptr)][col]    

    return ridge

# generate ice-rock boundary using hidden markov model(Viterbi) and 
# human feedback
# This function takes edge strength map and image array as input and  
# returns list of rows containining boundary points for ice-rock 
# @param: edge_strength: Gives us a derviative of image. Higher the value
# in edge strength, more likely a point is to be a part of a boundary
# @param: image: Image array
# @param: row_coord: Row number input by human for ice-rock boundary
# @param: col_coord: Column number input by human for ice-rock boundary
# @param: air_ice: list of rows for air-ice boundary
# using viterbi algorithm, we find the ice-rock boundary points. We consider rows 
# below 10px from the air-ice boundary points
# We run viterbi in two directions, 1.) from col_coord till all we reach the 
# end of columns. 2.) From col_coord to the start of columns.
# We consider minimum cost instead of maximum probability to avoid underflow issues
def get_feedback_rows_ice_rock(edge_strength,image, row_coord, col_coord, air_ice):
    edge_strength = edge_strength+1
    nrows = edge_strength.shape[0]
    ncols = edge_strength.shape[1]
    emission_probs = edge_strength/sum(edge_strength, axis = 0)
    emission_probs = -np.log(emission_probs)
    #emission_probs = -np.log(get_emission(image, edge_strength))

    #create viterbi table
    V_table = np.ones((edge_strength.shape[0], edge_strength.shape[1]))*np.Infinity
    for row in range(int(air_ice[0])+10, edge_strength.shape[0]):
        V_table[row,0] = -np.log(edge_strength[row][0]/np.sum(edge_strength[:,0])) + emission_probs[row,0]

    V_table[:,col_coord] = np.Infinity
    V_table[row_coord,col_coord] = -np.log(1)
    min_state = np.zeros((nrows, ncols))
            
    
    
    for t in range(col_coord+1, edge_strength.shape[1]):
        for row in range(int(air_ice[t])+10, edge_strength.shape[0]):
            min_cost = np.Infinity
            for r1 in range(edge_strength.shape[0]):

                distance_distr = get_transition_prob(row, r1)
                transition = distance_distr
                val = V_table[r1][t-1]+transition 
                if(val < min_cost):
                    min_cost = val
                    min_state[row][t] = r1
                    V_table[row][t] = min_cost + emission_probs[row][t]

    for t in range(col_coord-1, 0,-1):
        for row in range(nrows):
            min_cost = np.Infinity
            for r1 in range(int(air_ice[t])+10, nrows):

                distance_distr = get_transition_prob(row, r1)
                transition = distance_distr
                val = V_table[r1][t+1]+transition 
                if(val < min_cost):
                    min_cost = val
                    min_state[row][t] = r1
                    V_table[row][t] = min_cost + emission_probs[row][t]
    backptr = np.argmin(V_table[:,-1])
    ridge = np.zeros(ncols)
    for col in range(ncols-1, -1, -1):
        ridge[col] = int(backptr)
        backptr = min_state[int(backptr)][col]
    return ridge

# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))
    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    simple_bayes = get_bayes_rows(edge_strength)
    hmm_bayes = get_hmm_rows(image_array,edge_strength)
    imageio.imwrite('edges.png', uint8(255 * edge_strength / (amax(edge_strength))))

    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.

    airice_simple = simple_bayes[0]
    airice_hmm = hmm_bayes[0]
    airice_feedback= get_feedback_rows_air_ice(edge_strength,image_array,gt_airice[1], gt_airice[0] )


    icerock_simple = simple_bayes[1]
    icerock_hmm = hmm_bayes[1]
    icerock_feedback= get_feedback_rows_ice_rock(edge_strength,image_array, gt_icerock[1],gt_icerock[0],airice_feedback)

    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")
