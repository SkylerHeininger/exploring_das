# Breakdown of Dialogue act results, thoughts, and next steps

## Initial Results

### Speaker Dialogue ACts Comparisons

To start, we compare the speakers based on their dialogue act frequencies. This we expect to have high differences,
as the therapist and patient take much different roles in the conversation. 

![patient-therapist comparison](group_output/Speaker%20Comparison%20of%20DA%20groups_da_distribution.png "Speaker Comparison")

Here, we see the therapist asking far more answers and hedges from the patient, along with more questions from the therapist.


### Importance DA comparisons

![importance comparison](group_output/Importance%20Comparison%20of%20DA%20groups_da_distribution.png "Importance Comparison")

When comparing importance, we see several key relationships. In the important turns, there are fewer with essentially all dialogue act classes other than statements and hedging (which in terms of frequency there are more of these in non-important turns, there are just more of these).

### Importance DA comparisons across speakers

![patient importance comparison](group_output/DA%20Groups%20over%20Patient%20Speaking_da_distribution.png "Patient Importance Comparison")

Here, we see that really only hedging is important. 

![therapist importance comparison](group_output/DA%20Groups%20over%20Therapist%20Speaking_da_distribution.png "Patient Importance Comparison")

Here we see some more statistical significances, but nothing really important. Most are the unimportant turns are of higher frequencies than important turns, which is useful, but not amazing.

### Experiments so far

This includes the above comparisons, along withs some cross-speaker and cross-importance comparisons. Overall, comparing the DAs between speakers has the most significant / easily explanable features.

### Next steps

- Look at important turns more manually and see what is happening - are ther any patterns, etc?
- Check across therapists, to see how different different therapists communicate
- Check the train test split and see how this differs



