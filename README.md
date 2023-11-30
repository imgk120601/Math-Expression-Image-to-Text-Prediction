# Math-Expression-to-LaTeX-Code-Generation-Image-to-Text-Prediction-

1
1 Problem Statement
You are given two datasets of images which contain mathematical expressions and its
corresponding latex formula. The first dataset contains handwritten mathematical expression
in the images and the second contains the images which showcase mathematical
expressions that have been generated from LaTeX-based code(synthetic dataset).You are
provided with both train and test set for both the datasets. Your task is to train an ML
model that takes the image of the mathematical expression into input and outputs the
corresponding latex code.
![image](https://github.com/imgk120601/Math-Expression-to-LaTeX-Code-Generation-Image-to-Text-Prediction-/assets/64717239/15ec82a1-d475-48ae-817b-1b38946fe544)

Figure 1: Model architecture
2 Non-competitive part (40 marks)
In this part,you will be using an Encoder-Decoder architecture for modelling this
problem. An encoder is used to encode the given input into a context vector which is
further used for decoding by the decoder. The decoder is applied on this context vector
to generate the sequence auto-regressively(one word/character at a time). Your task is to
implement an encoder which will take the image of the mathematical expression as input
and the decoder which will output the latex formula for the provided expression.
You have to implement this part of the problem in two subparts:
2
• Part-a You should only use the synthetic training dataset for your training and report
the BLEU scores on validation set of handwritten and both test and validation
of the synthetic dataset.
• Part-b You should train your model on the synthetic dataset and then finetune the
same trained model on the handwritten dataset. Report the BLEU scores of the
model on validation set of handwritten and both test and validation of the synthetic
dataset.
You can use the following as a starting point:
1. Encoder: In this part you have to implement a simple CNN which takes as input
an image and returns a context vector to be used by the Decoder. Make sure to
resize the image to (224, 224) and normalise it.
• CONV1: Kernel Size → 5x5, Input Channels → 3, Output Channels → 32
• POOL1: Kernel Size → 2x2
• CONV2: Kernel Size → 5x5, Input Channels → 32, Output Channels → 64
• POOL2: Kernel Size → 2x2
• CONV3: Kernel Size → 5x5, Input Channels → 64, Output Channels → 128
• POOL3: Kernel Size → 2x2
• CONV4: Kernel Size → 5x5, Input Channels → 128, Output Channels → 256
• POOL4: Kernel Size → 2x2
• CONV5: Kernel Size → 5x5, Input Channels → 256, Output Channels → 512
• POOL5: Kernel Size → 2x2
• AvgPool2D: Window Size → 3x3 (Output size : 1x1x512)
Use ReLU as the activation function for all the layers apart from the Pooling layers.
For all Pool and Conv operations use the default size with no zero padding.
2. Decoder: Use a single layer LSTM as the architecture of choice that takes the
context vector as input and generates the latex formula. Set the dimensions of
LSTM class of pytorch with the following:
• Embedding Layer: → 512 (A learnable embedding for the output vocabulary)
• Hidden Layer: → 512 dimensions
• Output Layer: → Output Vocabulary size; transforms the hidden representation
into the vocabulary space.
You will first have to create a vocabulary from the formulas in the training dataset
and then initialise an embedding for each word/character in your vocabulary. Since
each formula can be of varying length, use padding to make all list sizes consistent.
3
3. Training strategy: Use cross-entropy as your loss function and teacher-forcing for
training the decoder. Don’t forget to use a START and END token to allow for
variable length formula generation from the decoder. When passing input to the
LSTM cell, you need to concatenate the context vector with the learned embedding
of the output of the previous timestep. This can be done in two phases:
• First, you can use teacher forcing where you will concatenate the context vector
with the embedding of the ground truth label of the previous timestep.
• Second, you can use the learned embedding of the output of the previous
timestep and concatenate it with the context vector and treat it as the input
to the current timestep.
• Use teacher forcing for 50% of the time during training.
• Hint: Embedding Class of pytorch can be used to maintain and learn the
embeddings of labels in the output vocab.
4. Metric Scores You will be graded on BLEU score. You can read about it in
Section 5.3 of this pdf document.
