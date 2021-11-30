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


# generate transition probabilities for a row to all rows in next column
# This function takes row of column 1 and number of rows as input and returns the inverse 
# exponential distance probability
# @param: row: Row of current column
# @param: len: Total number of rows in one column
def get_distance_distr(row1, len):
    diff =  [] 
    dists = []
    for row in range(0,edge_strength.shape[0]):
        dist = abs(row1-row)+1
       
        #diff.append(1/(2**dist+1))
        if dist <=6:
            diff.append(1/((dist+1)))
        else:
            diff.append(1/(exp(dist)))
        # if dist <=3:
        #     diff.append(dist+1)
        # else:
        #     diff.append(2**dist)
        #diff.append(dist*log(dist))
        dists.append(dist)

        """
        Some other distance distributions tried:
        1.) diff.append((len - dist)/len)
        2.) diff.append(1/(dist+1))
        """
    diff = diff/sum(diff)
    #print(f'diff: {diff}')
    #logdist = []
    #print(len(dists))
    # for i in range(edge_strength.shape[0]):
    #     #if dists[i] <=2:
    #         logdist.append(-diff[i]*np.log(diff[i]))
    #     # else:
    #     #     logdist.append(10000)
    return diff


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
    #new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    #new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)

# generate air-ice and ice-rock boundary using simple bayes
# This function takes edge strength map as input and  
# returns list of rows containining boundary points for air-ice and ice-rock
# @param: edge_strength: Gives us a derviative of image. Higher the value
# in edge strength, more likely a point is to be a part of a boundary.
# We calculate the 
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

# 
def get_hmm_rows(edge_strength):
    edge_strength = edge_strength+1
    ncols = edge_strength.shape[1]
    nrows = edge_strength.shape[0]

    emission_probs = -np.log(edge_strength/np.sum(edge_strength, axis = 0))
    V_table = np.ones((nrows, ncols))
    V_table[:,0] = emission_probs[:,0]

    # # calculation for viterbi table 
    # as minimum cost function

    # Calculation for air ice boundary
    for t in range(1, edge_strength.shape[1]):
        for row in range(edge_strength.shape[0]):
            distance_distr = get_distance_distr(row, edge_strength.shape[0])
            transition = -log(distance_distr)
            V_table[row,t] = min(V_table[:,t-1]+transition+emission_probs[row, t])
            
    ridge = V_table.argmin(axis=0)

    # # Calculation for ice-rock boundary

    V_table_ir = np.ones((nrows, ncols))*100000
    median_air_ice_row = int(median(ridge))
    for row in range(median_air_ice_row+10, edge_strength.shape[0]):
        V_table_ir[row,0] = -np.log(edge_strength[row][0]/np.sum(edge_strength[:,0]))

    for t in range(1, edge_strength.shape[1]):
            for row in range(median_air_ice_row+10, edge_strength.shape[0]):
                distance_distr = get_distance_distr(row, edge_strength.shape[0])
                #transition = distance_distr
                transition = -log(distance_distr)
                #print(f'row: {row}, dist: {distance_distr}, transition: {transition}')

                V_table_ir[row, t] = min(V_table_ir[:,t-1]+transition+emission_probs[row, t])
    ridge1 = V_table_ir.argmin(axis=0)
    return (ridge,ridge1)

def get_feedback_rows_air_ice(edge_strength, row_coord, col_coord):
    edge_strength = edge_strength+1

    emission_probs = edge_strength/sum(edge_strength, axis = 0)
    emission_probs = -np.log(emission_probs)

    #create viterbi table
    V_table = np.ones((edge_strength.shape[0], edge_strength.shape[1]))
    V_table[:,0] = emission_probs[:,0]
    V_table[:,col_coord] = emission_probs[:,col_coord]
    V_table[row_coord,col_coord] = -np.log(1)

    for t in range(col_coord-1, 0,-1):
        for row in range(edge_strength.shape[0]-1,-1, -1):
            distance_distr = get_distance_distr(row, edge_strength.shape[0])
            transition = -log(distance_distr)
            V_table[row,t] = min(V_table[:,t+1]+transition+emission_probs[row, t])
    
    for t in range(col_coord+1, edge_strength.shape[1]):
        for row in range(edge_strength.shape[0]):
            distance_distr = get_distance_distr(row, edge_strength.shape[0])
            transition = -log(distance_distr)
            V_table[row,t] = min(V_table[:,t-1]+transition+emission_probs[row, t])
            
    ridge = []
    ridge = np.argmin(V_table, axis = 0)
    

    return ridge

def get_feedback_rows_ice_rock(edge_strength, row_coord, col_coord, air_ice):
    edge_strength = edge_strength+1

    emission_probs = edge_strength/sum(edge_strength, axis = 0)
    emission_probs = -np.log(emission_probs)

    #create viterbi table
    V_table = np.ones((edge_strength.shape[0], edge_strength.shape[1]))*10000
    V_table[:,0] = emission_probs[:,0]
    V_table[:,col_coord] = emission_probs[:,col_coord]
    V_table[row_coord,col_coord] = -np.log(1)
    median_airice = int(median(air_ice))

    for t in range(col_coord-1, 0,-1):
        for row in range(edge_strength.shape[0]-1, median_airice+10, -1):
            distance_distr = get_distance_distr(row, edge_strength.shape[0])
            transition = -log(distance_distr)
            V_table[row,t] = min(V_table[:,t+1]+transition+emission_probs[row, t])
    
    for t in range(col_coord+1, edge_strength.shape[1]):
        for row in range(median_airice+10, edge_strength.shape[0]):
            distance_distr = get_distance_distr(row, edge_strength.shape[0])
            transition = -log(distance_distr)
            V_table[row,t] = min(V_table[:,t-1]+transition+emission_probs[row, t])
            
    ridge = []
    ridge = np.argmin(V_table, axis = 0)
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
    hmm_bayes = get_hmm_rows(edge_strength)
    imageio.imwrite('edges.png', uint8(255 * edge_strength / (amax(edge_strength))))

    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.

    airice_simple = simple_bayes[0]
    airice_hmm = hmm_bayes[0]
    airice_feedback= get_feedback_rows_air_ice(edge_strength,gt_airice[1], gt_airice[0] )


    icerock_simple = simple_bayes[1]
    icerock_hmm = hmm_bayes[1]
    icerock_feedback= get_feedback_rows_ice_rock(edge_strength, gt_icerock[1],gt_icerock[0],airice_feedback)

    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")
