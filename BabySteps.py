from sklearn import tree
#Test Program
#As we are only doing for 2 fruits we convert all data to numeric

#Features are its properties
#FOr simplicity we consider only 2 (Rough/Smooth) --> 0/1
#The other feature is the weight
features = [[140, 1],[130, 1],[150, 0],[160, 0],[120, 1],[120, 0]]

#Labels 0 --> APple 1 --> Orange
labels =  [0,0,1,1,1,0]

#Classifier is the one which decides what is what
#Its like a box of rules
#Our classifier is a decision tree
#The input and output is the same
#Right now this box is empty and we need to train it
clf = tree.DecisionTreeClassifier()

#Learning algorithm is the procedure that creates thes rules
#fit is a built in trained classifier
clf = clf.fit(features,labels)



