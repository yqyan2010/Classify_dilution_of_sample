# Classify_dilution_of_sample
It classifies testing samples as being diluted or not being diluted

## Background
Diluted sample has fake large test values. For example, assume real mercury (Hg) level is 0.001 ppm, however, if the sample is diluted 1000 times, the Hg level will be calculated as 1 ppm, such high level is fake. In practice, our lab relies on people to manually filter out diluted sample, therefore, preventing fake values being submitted as test results.

## Goal
The classification model uses machine learning to identify report as two categories: the report that comes from a dilute sample versus the report that comes from a non-diluted sample. By using this, the algorithm will send alert message if a diluted sample report is selected to report fake large values. 

## Models in use
