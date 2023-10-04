# -*- coding: utf-8 -*-
"""
Author: Grace Milner

This script uses an Agent-Based Model to predict patterns of urban expansion in Bahir Dar, Ethiopia by 2033.

The script uses population growth estimates and input data regarding suitability 
for different types of urban expansion, as well as neighbourhood functions and ranking functions
to determine most likely expansion patterns guided by most probable agent decision making.

The process is carried out in one-year planning episodes over a ten-year period.
Complexities such as reallocation and policy prioritisation are taken into account.

The output is a land cover prediction map. 


"""

# Loading required libraries and functions
import numpy as np
import scipy.stats as ss
import os  
import matplotlib.pyplot as plt

# Defining working directory
os.chdir(r'C:\VUB\Year 2\ModLUC\Exercise 3\new_data')

import Import_files as Im

# Defining a function to normalise the imported arrays between 0 and 1, for a meaningful relative value
# (value - min / max - min)
def Norm(array):
    norm_array = (array-np.min(array))/(np.max(array)-np.min(array))
    return norm_array
    

# Importing Land Cover (LC) map from 2023 as starting point for model
LC = Im.rst("LC_(un)planned.tif")

# Normalised variable rasters to use in calculations
dist_cent = Norm(Im.rst("Distance_to_centers_EX3.tif"))
areas_int = Im.rst("Areas_of_interest_Updated.tif") # don't need to normalise this, already 0-1. 
dist_invest = Norm(Im.rst("Investment_difficulty_map.tif"))
dist_local_r = Norm(Im.rst("Distance_to_localroads_Updated.tif"))
dist_prim_r = Norm(Im.rst("Distance_to_primaryroads_EX3.tif"))
dist_riv = Norm(Im.rst("Distance_to_rivers_EX3.tif"))
dist_wor = Norm(Im.rst("Distance_to_worshiping_EX3.tif"))
inf_suit = Norm(Im.rst("Infrastructure suitability_Updated.tif"))
slope = Norm(Im.rst("Slope_EX3.tif"))

# Additional Land Cover info 
land_cover_info = Im.rst("gee_LC_new.tif")
#get unique values and counts of each value
unique, counts = np.unique(land_cover_info, return_counts=True)
#display unique values and counts side by side
print("Land cover pixel counts in 2023")
print(np.asarray((unique, counts)).T)



# Defining neighbourhood function 
    # aim is to find cells nearest to existing unplanned cells where people could move to (so near unplanned but not already planned)

def NH(array):
    # setting up new array to receive values, initialised with zeros
    x,y=array.shape
    NH=np.zeros((x,y))
    # non-borders
    for i in np.arange(1,x-1):  # (np.arange returns evenly spaced values between intervals)
        for j in np.arange(1,y-1):
            mask=array[i-1:i+2,j-1:j+2] # creates window around an individual cell
            NH[i,j]=(np.sum([mask])-mask[1,1])/8  # sums all cells in window, removes the central cell, divides by 8 (number of cells) to give average 
          
    # fill borders
    NH[0,:]=NH[1,:]
    NH[x-1,:]=NH[x-2,:]
    NH[:,0]=NH[:,1]
    NH[:,y-1]=NH[:,y-2]
    
    # finishing
    return(NH)

# Defining ranking function
def rank_array(array):
    # no need to flatten first, because ss.rankdata default flattens the array anyway
    ranks = ss.rankdata(-array, method = 'ordinal') 
        # Here (above), calculating rank of pixels 
        # Ordinal means pixels with same probability still given individual ranks
        # Using the negative of the array (multiplying by -1) means the ranks are highest to lowest (largest probability given rank '1')
    return(ranks) #gives the 1D array of pixel ranks
    
# Defining Top Rank function
def Top_Rank(Array, N):
    ranks = rank_array(Array) #ranking the pixels largest to smallest, output 1D array
    TopRank = ranks.copy() #creating copy of ranks array
    TopRank[TopRank > N] = -1 #if rank higher than required pixel number (N), changed to -1
    return(TopRank)





#%%

"""
0= no data
1= planned
2= unplanned
5= expansion area
"""
# Define the population variables
# Assumes population proportions between unplanned and planned stay constant. 
# Total population doubles every 10 years (so in this investigation, doubles once)
P = 250000 #population
planned_P = P * 0.45 #(45% of population comes from planned pixels)
unplanned_P = P * 0.55 #(55% of population comes from unplanned pixels)

# Define (copy) land cover map
LC_new2D = LC  #assigning to new variable to not change original data

# Set steps to start a while loop for every planning episode
planning_step = 1
new_pop = P
total_moved_unplanned = 0
         
# Start a while loop      
while planning_step < 11: #(so each 'planning step' is one year, up to max ten years)
    ' Prep steps '     
    # apply the neigbourhood function on the unplanned pixels.
    # (the output from the neighbourhood function is a variable which gets used in the utility function of the unplanned pixels)
    unplanned_NH = NH(LC_new2D==2)
    
    # Apply the coefficients for planned and unplanned areas and formulate the utility functions for both planned and unplanned pixels
    # Making variables with 'distance to' logic negative
    unplanned_utility = (areas_int*0.24 - dist_riv*0.152 - dist_wor*0.128 + unplanned_NH*0.236 - dist_local_r*0.244) *1000 
    planned_utility = (slope*-0.144 - dist_cent*0.214 - dist_prim_r*0.156 - inf_suit*0.244 - dist_invest*0.242) *1000 #(multiplying by 1000 to make values valid for further calculation)
 
# TESTING
# nrow, ncol = LC_new2D.shape # original shape
# test= np.reshape(unplanned_assigned, (nrow, ncol))

# plt.imshow(test)  
  
    ' Population calculations'
    # calculate the additional population (net population growth)
    extra_P = P/10 #(P will double in ten years, so in one year the growth must be P/10 assuming constant growth rate)
    
    # calculate the number of people living in a planned pixel
    planned_P_density = planned_P / np.count_nonzero(LC==1) # dividing total planned pop by number of planned pixels
    
    # calculate number of people living in unplanned pixel
    unplanned_P_density = unplanned_P / np.count_nonzero(LC==2) # dividing total unplanned pop by number of unplanned pixels
    
    # calculate increase in planned pixels (for each time step)
    extra_planned_P = extra_P * 0.45  # proportion of new population that will be in planned pixels
    extra_planned_pixels = np.round(extra_planned_P / planned_P_density) # how many new pixels needed to accommodate that increased planned pop (rounded)
    
    # calculate increase unplanned pixels (for each time step)
    extra_unplanned_P = extra_P * 0.55  # proportion of new population that will be in unplanned pixels
    extra_unplanned_pixels = np.round(extra_unplanned_P / unplanned_P_density)  # how many new pixels needed to accommodate that increased unplanned pop (rounded)

    
    'STEP 1: assign the new planned pixels (they can take over unplanned pixels, hence go first) ' 
       # make the arrays 1D
    LC_new1D = LC_new2D.flatten()
    new_planned_utility_1D = (np.copy(planned_utility)).flatten()  # making copy to not change original data
    
       # make utility -1000 where no planners can come 
       #(blocked existing planned cells as they are not moved, restricted to only unplanned or expansion cells)      
    new_planned_utility_1D[LC_new1D==1] = -1000 # assigning -1000 in new utility array in cells where already planned
    new_planned_utility_1D[LC_new1D==0] = -1000 # assigning -1000 in new utility array in cells where no data (not valid for expansion)
    
       # calculate ranks for planned pixels 
    planned_ranks = Top_Rank(new_planned_utility_1D, extra_planned_pixels)
       
       # assign the value 1 for the new planned pixels in the array
    planned_assigned= np.where(planned_ranks>0, 1, 0) #(where planned ranks bigger than 0, give value 1, otherwise 0)
    new_LC = np.copy(LC_new1D) #making copy of original LC
    new_LC[planned_assigned==1] = 1 #changing cell values in new general LU map
       
    ' Step 2: How many extra unplanned pixels are relocated? '
        #calculating difference
    moved_unplanned = (np.count_nonzero(LC_new1D==2)) - (np.count_nonzero(new_LC==2))  # old total minus new total
        # And based on that calculate how many pixels are needed for the unplanned now, combined with the original demande based on the population growth rate
    new_unplanned_pixels = extra_unplanned_pixels + moved_unplanned
    
    
    ' Step 3: Allocate those new unplanned pixels '
        # make the array 1D
    new_unplanned_utility_1D = (np.copy(unplanned_utility)).flatten()  # making copy to not change original data
       # make utility -100 where no unplanned pixels can come
    new_unplanned_utility_1D[new_LC==1] = -1000 # assigning -1000 in cells where already planned (including recently added within same loop)
    new_unplanned_utility_1D[LC_new1D==0] = -1000 # assigning -1000 in new utility array in cells where no data (not valid for expansion)
    new_unplanned_utility_1D[new_LC==2] = -1000 # assigning -1000 in new utility array in cells where already unplanned

    
       # calculate ranks for unplanned pixels 
    unplanned_ranks = Top_Rank(new_unplanned_utility_1D, new_unplanned_pixels)
       
       # assign the value 2 for the new unplanned pixels in the array
    unplanned_assigned= np.where(unplanned_ranks>0, 2, 0) #(where planned ranks bigger than 0, give value 2, otherwise 0)
       #new_LC = np.copy(LC_new1D) #making copy of original LC (maybe not needed)
    new_LC[unplanned_assigned==2] = 2 #changing cell values in new general LU map

    ' Step 4: Prepare for next loop '
    # Calculate the new population number
    new_pop = P + (extra_P*planning_step) # original population, plus yearly extra * number of years
    # Reshape and assign the new values (un)planned land cover map
    nrow, ncol = LC_new2D.shape # original shape
    LC_new2D = np.reshape(new_LC, (nrow, ncol)) #reshaping new LC map to 2D and reassigning to original variable ready for next loop
    # Print results from loop
    total_moved_unplanned += moved_unplanned # to keep track of how many unplanned pixels moved in total
    print("Year " + str(2023 + planning_step)+ ":")
    print("Total population: " + str(new_pop))
    print("Total unplanned pixels moved = " + str(total_moved_unplanned))
    print("Total unplanned population moved = " + str(total_moved_unplanned*unplanned_P_density))
    # Count the steps
    planning_step += 1

         
#%% step 4: export

#import gdal # other way to import gdal
# from osgeo import gdal # to work with spatial data. Python 3.4 or lower needed
# import Export_files
# Example=gdal.Open('LC_(un)planned.tif')
# Path_Name= 'LC_2033.tif' #where to write the file + name
# Export_files.array(LC_new2D,Example,Path_Name)
# plt.imshow(LC_new2D)


# Testing total number of pixels changed
old_planned = (np.count_nonzero(LC==1))
new_planned = (np.count_nonzero(LC_new2D==1))
old_unplanned = (np.count_nonzero(LC==2))
new_unplanned = (np.count_nonzero(LC_new2D==2))

print("Old planned pixels = " + str(old_planned))
print("New planned pixels = " + str(new_planned))
print("Old unplanned pixels = " + str(old_unplanned))
print("New unplanned pixels = " + str(new_unplanned))

# Calculating Land Cover change
land_cover_info[LC_new2D==1] = 0 # where the generated LC map has planned settlements, make pixel 0 in other LC map
land_cover_info[LC_new2D==2] = 0 # where the generated LC map has unplanned settlements, make pixel 0 in other LC map
unique2, counts2 = np.unique(land_cover_info, return_counts=True)
print("Land cover pixel counts in 2033")
print(np.asarray((unique2, counts2)).T)
counts_dif = counts - counts2
counts_dif_perc = np.round((counts_dif / counts) * 100)
print(np.asarray((unique2, counts_dif_perc)).T)
