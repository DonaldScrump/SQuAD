# Above all

This part of codes includes all that is needed to set up the program, with all data left out.
Histograms are just for illustration.

# Programming Assignment 4
Welcome to CS224N Project Assignment 4 Reading Comprehension.
The project has several dependencies that have to be satisfied before running the code. You can install them using your preferred method -- we list here the names of the packages using `pip`.

# Requirements

The starter code provided presupposes a working installation of Python 2.7, as well as a TensorFlow 1.1.0(rc2).

It should also install all needed dependencies through
`pip install -r requirements.txt`.

# Running your assignment

You can get started by downloading the datasets and doing dome basic preprocessing:

$ code/get_started.sh

Note that you will always want to run your code from this assignment directory, not the code directory, like so:

$ python code/train.py train

This ensures that any files created in the process don't pollute the code directoy.

# How to run the training program?

If you want to train the file automatically, possibly reusing previous weights in /train directory, run:

$ python code/train.py train

Or if you want to manually load the previous saved parameters, type:

$ python code/train.py load -p DIR_PATH