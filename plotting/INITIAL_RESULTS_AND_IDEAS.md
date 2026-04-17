# Breakdown of Dialogue act results, thoughts, and next steps

## Initial Results

### Speaker Dialogue ACts Comparisons

To start, we compare the speakers based on their dialogue act frequencies. This we expect to have high differences,
as the therapist and patient take much different roles in the conversation. 

![patient-therapist comparison](group_output/Speaker%20Comparison%20of%20DA%20groups_da_distribution.png "Speaker Comparison")

Here, we see the therapist asking far more quesitons, with more hedges and answers from the patient.


### Importance DA comparisons

![importance comparison](group_output/Importance%20Comparison%20of%20DA%20groups_da_distribution.png "Importance Comparison")

When comparing importance, we see several key relationships. In the important turns, there are fewer with essentially all dialogue act classes other than statements and hedging (which in terms of frequency there are more of these in non-important turns, there are just more of these).

### Importance DA comparisons across speakers

![patient importance comparison](group_output/DA%20Groups%20over%20Patient%20Speaking_da_distribution.png "Patient Importance Comparison")

Here, we see that really only hedging is important. 

![therapist importance comparison](group_output/DA%20Groups%20over%20Therapist%20Speaking_da_distribution.png "Patient Importance Comparison")

Here we see some more statistical significances, but nothing really important. Most are the unimportant turns are of higher frequencies than important turns, which is useful, but not amazing.

### Experiments so far

Summary: Frequency analysis across DAs in all comparisons, bag of words (BOW) of ngrams (2,3,4,5 grams or sequences of DAs) and graph-level analysis of DAs. Comparisons include: Across patient, therapist, nonimportance (sometimes combined patient/therapist importance), across speaker in conversation, across transcripts (some approaches have this, but really the across therapist should be more important). 

This includes the above comparisons, along withs some cross-speaker and cross-importance comparisons. Overall, comparing the DAs between speakers has the most significant / easily explanable features.

`graph_file_da.py` now includes a tool for manually rendering the dialogue acts in conversation, specifically surrounding important turns, making it easier to see what is happening within the conversations.

`get_question_response_patterns.py` looks at the run lengths for responses of each speaker when the patient and therapist ask questions. For example, how does the run length differ when patient asks therapist a question vs therapist asks patient a quesiton. This is done over patient and therapist importance separately, since there seem to be differences in labeling patterns between patient and therapist. Tihs file also looked at the DA's across codes, seeing differing distributions across different codes. This somewhat suggests we may need to group some codes or make a classifier that identifies different codes (harder but may be needed).

`view_common_patterns.py` attempts to find different common patterns in important-labeled turns, compared across codes. This has an important function of viewing patterns of DAs across a variety of ngrams and granularities.
- ngram: the number of items to view at once: If this is 2, each node in the graph is one DA, so an edge signifies connecting the two. If this is 2 or higher, a connection between two DAs will be represented within the node itself, and connections between nodes signify a 3 or higher connection. 
    - Ex: DAs: Q to S to A to S to ... (rest of conversation)
1gram: Each node contains 1 DA: Q -> S -> A -> S -> ...
2gram: Each node contains 2 DAs: (Q,S) -> (S,A) -> (A,S) -> (S,...)
3gram: Each node contains 3 DAs: (Q,S,A) -> (A,S,...)

And so on. This begs the question of: What is the best ngram to use for our case? This is also accompanied by the issue 
of granularity. Currently, we have two granularities: Raw DA classes, and groups of classes. For the latter, we previously used statements, canonical and non-canonical questions and answers, hedging, and backchannels. This was helpful for initially viewing the data, but to maximize transitions between different classes, we group all questions and answers together, and group other classes together too, so all are

Currently running this script using this:
`python .\common_patterns.py --dir ..\..\AC_output_csvs\ --graph_order 3 --min_edge_weight 3`

This also seems to benefit from getting the run length: use --bucketed runs, which modifies nodes to be either short, medium, long. This needs to be looked at more to see what best quantifies a short medium or long run.

Notes from graph-based methods:
- Using ngrams with higher than 2 graph order seems to make overly specific sequences, giving very few nodes and edges within the graph. 
- The binning seems to be a good choice, giving different weights/nodes to different lengths of DAs. This is especially important for the statements, as these have large run lengths, especially in certain codes.
- Switching to a uni-directed graph may be more clear for this - if things go back and forth, like QSAS, this leads to not going back and forth in the graph.

`python .\unidirectional_common_patterns.py --dir ..\..\AC_output_csvs\ --min_edge_weight 2 --include_context --context_window 5`

The context window of 5 seems to help a bit as a lot of times questions are lost in the therapy windows. The unidirectional approach seems helpful overall.


`python .\ngram_bow.py --dir ..\..\AC_output_csvs\ --min_edge_weight 2 --include_context --context_window 5`

This uses bag of words approach with ngrams to build dictionaries of common patterns. A "word" in this case would be a sequence of DA groups: statements, answers, etc. This approach also generally agrees with the graph-based approach, although I think the graph based approach may overall be better, as between ngrams there is repitition, and I think describing a code/importance is a better approach using a graph (connectivity, etc), rather than the bag of words.

### Next steps

- Try canonical/noncanonical questions and answers in the given experiments, use the frequency data to try and motivate this.
- Check the train test split and see how this differs



