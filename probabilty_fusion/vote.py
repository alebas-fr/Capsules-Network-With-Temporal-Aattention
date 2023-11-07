import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class VoteClassifier():
    """
    Implementation of vote Classifier for multimodal integration
    """
    def __init__(self,numbers_of_classifiers,number_of_classes,names_of_classifiers=None):
        """
        numbers_of_classifiers: Number of Model that we have (in our case it will be a number between 1 and 4)
        shape_of_classifiers: It correspond at the output of the model in our case we must have a probability distribution 
        names_of_classifiers: Correspond to names of classifier it's just to know which classifier correspond and have names instead of numbers
        values: All the values of classifier
        """
        self.numbers_of_classifiers = numbers_of_classifiers
        self.names_of_classifiers = names_of_classifiers
        self.number_of_values = None
        self.number_of_classes =  number_of_classes
        self.values = None
        
    def set_values(self,number_of_values):
        """
        This functions create the tensor values. This function must be call before to add values in a classifier
        number_of_values
        """
        self.number_of_values = number_of_values
        self.values = torch.zeros(self.numbers_of_classifiers,self.number_of_values,self.number_of_classes)

    def add_values_in_classifier(self,classifier,values):
        """
        This function add all values of classifier into the attribute values 
        
        classifier: Number of classifier where we want to set the values
        values: All the values of the classifier, it must be a probability distribution 
        """
        if type(self.values) == type(None):
            raise TypeError("You must apply the function set_values before")
        self.values[classifier] = values
        
    def add_value_in_classifier(self,classifier,index,value):
        """
        This function adds one value of classifier into the attribute values 
        
        index: Index of value that we want to add
        classifier: Number of classifier where we want to set the values
        value: value of the classifier, it must be a probability distribution 
        """
        if type(self.values) == type(None):
            raise TypeError("You must apply the function set_values before")
        self.values[classifier,index] = value
    
    def compute_score_classification(self):
        """
        This function compute the probabilty for each prediction in values considering the results of each classifier.
        The formula to get the new probabilities is yt = 1 N ·∑ i Y (H(F (St,i)))
        return: The new probabilities considering all the classifier as torch Tensor of shape [1,shape_of_classifier]
        """
        new_probs = torch.zeros(self.number_of_values,self.number_of_classes)
        i = 0
        while i<self.number_of_values:
            prob = self.values[:,i]
            new_probs[i] = prob.sum(0)/self.numbers_of_classifiers
            i+=1
        return new_probs
