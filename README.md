kaggle-decmeg2014
=================

Code for the Kaggle DecMeg2014 competition


Requirements:

Data must be on a local file system with the following structure:

<root dir>
- /test
-- /test_subjectXX.mat files
- /train
-- /train_subjectXX.mat files

Data pipeline:
1) Run the munge file
- Set the in_dir to the <root dir> above
- Choose the number of components for the PCA reduction

This will create scaled csv's with 90K features plus a parallel set of csv's with # components features

2) Run the analyze file

- The evaluate() method is a playground for trying and tuning algorithms

- The train() and predict() methods called in that order will train the classifier on the appropriate trian data and then write a prediction file that can be loaded to Kaggle.


Notes:
- Did a simple run with 5000 components on PCA fed into a default Logistic Regression classifier.  Resulted in 43rd place with .65079 accuracy
-- Weird because x-val showed 70%+ accuracy... need to research

- Tried running on average of time series and average accross all sensors and ended up with ~50% on both

- Based on research paper noting that physioligical differences cause "feature shifts" betweek patients, PCA is a dead approach



To Do:
- try different PCA sizes
- figure out how to load all 90K features
- figure out which features are most important and whittle down
- figure out if different brains activate different sensors differently... i.e. how do features line up???
-- Read the PDF - discusses one way of dealing with feature sets that shift