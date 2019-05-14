# Word_embedding


Word Embedding Tutorial: word2vec using Gensim [EXAMPLE]

What is Word Embedding?

Word Embedding is a type of word representation that allows words with similar meaning to be understood by machine learning 
algorithms. Technically speaking, it is a mapping of words into vectors of real numbers using the neural network, probabilistic 
model, or dimension reduction on word co-occurrence matrix. It is language modeling and feature learning technique. 
Word embedding is a way to perform mapping using a neural network. 
There are various word embedding models available such as word2vec (Google), Glove (Stanford) and fastest (Facebook).

Word Embedding is also called as distributed semantic model or distributed represented or semantic vector space 
or vector space model. 

As you read these names, you come across the word semantic which means categorizing similar words together. 
For example fruits like apple, mango, banana should be placed close whereas books will be far away from these words. 
In a broader sense, 
word embedding will create the vector of fruits which will be placed far away from vector representation of books.

In this tutorial, you will learn

    What is Word Embedding?
    Where is Word Embedding used?
    What is word2vec?
    What word2vec does?
    Why Word2vec?
    Word2vec Architecture
    Continuous Bag of Words.
    Skip-Gram Model
    The relation between Word2vec and NLTK
    Activators and Word2Vec
    What is Gensim?
    Code Implementation of word2vec using Gensim

Where is Word Embedding used?

Word embedding helps in feature generation, document clustering, text classification, and natural language processing tasks. 
Let us list them and have some discussion on each of these applications.

    Compute similar words: Word embedding is used to suggest similar words to the word being subjected to the prediction model. 
    Along with that it also suggests dissimilar words, as well as most common words.
    Create a group of related words: It is used for semantic grouping which will group things of similar characteristic together 
    and dissimilar far away.
    Feature for text classification: Text is mapped into arrays of vectors which is fed to the model for training 
    as well as prediction. Text-based classifier models cannot be trained on the string, 
    so this will convert the text into machine trainable form. Further its features of building semantic help in 
    text-based classification.
    Document clustering is another application where word embedding is widely used
    Natural language processing: There are many applications where word embedding is useful and wins over 
    feature extraction phases such as parts of speech tagging, sentimental analysis, and syntactic analysis.

    Now we have got some knowledge of word embedding. Some light is also thrown on different models to implement word embedding. This whole tutorial is focused on one of the models (word2vec). 

What is word2vec?

Word2vec is the technique/model to produce word embedding for better word representation. 
It captures a large number of precise syntactic and semantic word relationship. 
It is a shallow two-layered neural network. Before going further, 
please see the difference between shallow and deep neural network:

The shallow neural network consists of the only a hidden layer between input and output whereas deep neural network contains 
multiple hidden layers between input and output. Input is subjected to nodes whereas the hidden layer, 
as well as the output layer, contains neurons.

word2vec is a two-layer network where there is input one hidden layer and output.

Word2vec was developed by a group of researcher headed by Tomas Mikolov at Google. 
Word2vec is better and more efficient that latent semantic analysis model.

What word2vec does?

Word2vec represents words in vector space representation. 
Words are represented in the form of vectors and placement is done in such a way that similar meaning words appear together 
and dissimilar words are located far away. This is also termed as a semantic relationship. 
Neural networks do not understand text instead they understand only numbers. 
Word Embedding provides a way to convert text to a numeric vector.

Word2vec reconstructs the linguistic context of words. Before going further let us understand, 
what is linguistic context? In general life scenario when we speak or write to communicate, other people try to figure out 
what is objective of the sentence. 
For example, "What is the temperature of India", here the context is the user wants to know "temperature of India" which is 
context. In short, the main objective of a sentence is context. 
Word or sentence surrounding spoken or written language (disclosure) helps in determining the meaning of context. 
Word2vec learns vector representation of words through the contexts.
Why Word2vec?
Before Word Embedding

It is important to know which approach is used before word embedding and what are its demerits and then we will move to the 
topic of how demerits are overcome by Word embedding using word2vec approach. 
Finally, we will move how word2vec works because it is important to understand it's working.
Approach for Latent Semantic Analysis

This is the approach which was used before word embedding. 
It used the concept of Bag of words where words are represented in the form of encoded vectors. 
It is a sparse vector representation where the dimension is equal to the size of vocabulary. If the word occurs in the dictionary, it is counted, else not. To understand more, please see the below program.

for reference:https://www.guru99.com/word-embedding-word2vec.html
