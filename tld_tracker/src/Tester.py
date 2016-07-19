######################################################################################################################################################################
#   Written for Computer Vision team on McGill Robotics by Alexander Chatron-Michaud & Tristan Struthers
#   Usage: 
#          - First command line argument is flag 1. If the model is to be scored, use the flag -test. Otherwise, for prediction, use the "full" flag -full
#          - Second command line arg is flag 2.  It is a filepath to the data path you want to predict values for
#
######################################################################################################################################################################

import sys, string, pickle, os, time
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn import preprocessing

X = []
X_cv = []
X_test = []
y_cv = []
y_test = []
NN_model = None

def addToX(textfile): #Opens "textfile" and adds line-by-line training vectors to X_train and gives them the file's name as it's y value
        global X, X_cv, X_test, y_cv, y_test, NN_model
        correct_val = os.path.splitext(textfile)[0]
        with open(textfile) as f:
                for x in f.readlines():
                        x = x.replace(',', '')
                        x = x.replace('(', '')
                        x = x.replace(')', '')
                        line = x.split()
                        outline = []
                        for char in line:
                                outline.append(float(char))
                        X.append(outline)
                f.close()

def setup(full):
    global X, X_cv, X_test, y_cv, y_test, NN_model
    NN_model = joblib.load('DATA/NN_model.pkl')
    if full:
		if len(sys.argv < 3) or not os.path.isfile(sys.argv[2]):
			print "Please supply a vaild filepath for the set you want to predict."
			exit()
		addToX(sys.argv[2])
    else: # -test cmd line flag was used
        X_cv = joblib.load('DATA/X_cv.pkl')
        X_test = joblib.load('DATA/X_test.pkl')
        y_cv = joblib.load('DATA/y_cv.pkl')
        y_test = joblib.load('DATA/y_test.pkl')

def main():
    global X, X_cv, X_test, y_cv, y_test, NN_model
    if len(sys.argv) < 2:
        print "Please use flag -full or -test when running the program"
        exit()
    if sys.argv[1] == "-full":
        full = True
    elif sys.argv[1] == "-test":
        full = False
    else:
        print "Please use flag -full or -test when running the program"
        exit()

    print "Setting up the model and loading data..."
    setup(full)
    print "Successfully loaded data!"

    if full:
    	for elem in X:
    		print NN_model.predict(elem)
    else:
    	print NN_model.score(X_cv, y_cv)
    	print NN_model.score(X_test, y_test)

main()
