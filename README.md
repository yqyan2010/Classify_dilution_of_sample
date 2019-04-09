# Dilution prediction

## Highlights this repository
* I write algorithm in this repository, rather than using built-in library from scikit-learn or keras.
* As of now, it contains logistic and neural network models.
* Development is in process.

## Goal
The data set contains ~1000 numericla testing data.Some data (~30%) correspond to samples that are diluted. The goal is to train a binary classification model to predict whether or not if a sample is diluted.

## Background
Diluted sample has fake large test values. For example, assume real mercury (Hg) level is 0.001 ppm, however, if the sample is diluted 1000 times, the Hg level will be calculated as 1 ppm, such high level is not true. In practice, our lab relies on people to manually filter out diluted results.The goal is to implement machine learning model to run filter before submitting results.

## Models in use
