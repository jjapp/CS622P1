# Project 1:  CS622
## John Appert
## 21 September 2020

## Decision Trees

### DT_train_binary

I implemented the algorithm by using a recursive function and stored the results in a class.  I did get stuck on this
portion of the exercise.  I also used the following tutorial 'https://www.youtube.com/watch?v=LDRbO9a6XPU' to get through
although I implemented my own version of the function.  I did use the same types of classes though and the build_tree
function is similar.  However, I implemented my version using numpy arrays and built different helper functions.


#### Helper Functions

* get_entropy:  returns entropy for a dataset
* get_max_gain_idx:  returns the numpy index for the feature that gives the best information gain.
* split_array:  splits the training data based on yes/no values.
* build_tree: does some munging and builds the actual decision tree.

#### Classes

* Partition: Stores the index we split on and it's value (yes or no).  Also provides a method to
compare an observed value with the decision point.
* Leaf:  Stores the prediction for a leaf
* DecisionNode: Stores a node.

### DT_make_prediction

In this class I do two things.  First, I check to see if I am at a leaf.  If so I return the
prediction for that leaf.  If not I compare the value in the node to my decision and then recursively
call the prediction class while working my way down the tree until I hit a leaf.

### DT_test_binary

I iterate through a dataset and compare the predicted results to the test results.  Accurate predictions
are added to one list and false predictions added to another.  I then calculate the percent of accurate
predictions.

### DT_train_real

### DT_test_real

## K Nearest Neighbor

#### KNN_test

* Loop through all rows in X_test
* Create a list to hold all distances and votes
* Nested loop through all rows in X_train
* Append distance and vote to holding list
* Sort the list of votes by distance using a bubble sort
* Slice the list at K
* Build two lists, one accurate classifications, one inaccurate.
* Use the sum of the two lists to calculate accuracy

#### choose_K

* Calculate number of rows in X_train
* Set Max K to number of rows
* Create variables to hold best accuracy and best K
* Loop through data using KNN_test and varying Ks
* Update best accuracy and best K based on results.
* Return best K.

## K_Means

### K_Means(X,K, mu)

#### Helper functions
* assign_to_cluster: assigns a point to the correct cluster
* get_centers: calculates new center points
* get_max_distance: gets the distances between two points

#### Implementation

* First check for whether or not the centers are empty.  If so create random points.
* Create a dictionary to hold each cluster
* Use the assign_to_cluster helper function to assign points to clusters
* Calculate new mu

### K_Means_better

* Loop through K_means until the distance between old mu and new mu is zero.
* return mu

