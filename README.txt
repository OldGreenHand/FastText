1. Minimal installation instructions to get started with Tensorflow on lab computers
	
1.1 Install on your laptops
Tensorflow provides details installation guide on https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html. Follow the one most suitable for your computer.
   1.2 Instructions for lab computers
a) Download installation script for Linux from http://conda.pydata.org/miniconda.html
b) Run the script by issuing the command "bash Miniconda2-latest-Linux-x86_64.sh" (this will start installation)
c) Press enter key, when prompted
d) Scroll down the Licence Agreement (using space key) and write yes at the promp asking if you agree with the terms
e) Press enter to continue installation at default location (location of install)
f) The installation will start downloading required packages
g) Write yes at promp asking for the script to add new packages to your path
h) Install numpy
	conda install numpy
i) Follow the instructions on  https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#using-conda to install tensorflow
j) Install nltk:
	conda install nltk


2. Hand-on experiences with NLTK
If you have never used NLTK, please follow the instructions in Section 3.1 of http://www.nltk.org/book/ch03.html to work on the example with Electronic books. If you have not installed NLTK before, you may need to install the tokenizer for news:
	import nltk
	nltk.download('punkt')

3. Tensorflow
	3.1 Follow the tutorial on 	https://www.tensorflow.org/versions/r0.10/tutorials/index.html to 	understand how to build a logistic regression with Tensorflow.

	3.2 Learn from the tutorial on 	https://www.tensorflow.org/versions/r0.10/tutorials/word2vec/index.html#ve	ctor-representations-of-words to understand how to initialize word 	embeddings and performing embedding lookup. Note that, you are not 	required to understand noise-contrastive estimation as well as 	visualization of embeddings.


4. Starter code fastText.py
The starter code will enormously reduce your efforts of coding. Try to complete the methods fastText.py and read the comments in fastText.py carefully.


5. Data
There are 5 data files for the purpose of training, validation, and testing. Each row in sentences_<XXX>.txt contains an instance (a news title), whose label(category) is in the corresponding row in labels_<XXX>.txt.
 
sentences_train.txt : sentences for training, one row per instance.
labels_train.txt : classification labels for training, one row per instance.
sentences_dev.txt : sentences for development (validation) set, one row per instance.
labels_dev.txt : classification labels for development (validation) set, one row per instance.
sentences_test.txt : sentences for testing, one row per instance.
You will be provided with the ground-truth of the dataset (labels_test.txt) in the NLP grading lab. 
