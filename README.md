# Dialogue Act Segmentation for Open Recordings

## Description of contents

The most important file within this folder is `daseg_pipeline.py`. This will load all files from a given directory, and perform dialogue act segmentation on the given column. Usage of this would be:
```
python daseg_pipeline.py --directory <data directory> --output_dir <location for saved files> --col_with_text <column containing text>
```

Please note that this may not work exactly as intended initially, as the data is in a different format. The primary change is that the original code assumes the data is structured at the word level (one word + any punctuation per row), but our data is not. I have made some initial changes that should work. The output from this will be at the word level.

### Graphing

In the `plotting` folder, there contains several scripts for some initial plots of the data.

`get_hists.py` will grab histograms for all files, and for the entire supplied directory. Useful for seeing what is happening in a directory.

`comparative_plotting.py` will compare two different directories across their frequencies.

`graph_file_da.py` is in-progress, with the purpose of exploring DAs throughout a transcript, rather than just at the frequency level.


## Resources

Labeling manual for dialogue acts: https://web.stanford.edu/~jurafsky/ws97/manual.august1.html

Zelasko paper: https://arxiv.org/pdf/2107.02294



