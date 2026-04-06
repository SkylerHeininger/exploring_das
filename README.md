# Dialogue Act Segmentation for Open Recordings

## Description of contents

The most important file within this folder is `daseg_pipeline.py`. This will load all files from a given directory, and perform dialogue act segmentation on the given column. Usage of this would be:
```
python daseg_pipeline.py --directory <data directory> --output_dir <location for saved files> --col_with_text <column containing text>
```

`daseg_pipeline.sh` uses this on Discovery, simply replace the CLI args in that!

Please note that this may not work exactly as intended initially, as the data is in a different format. The primary change is that the original code assumes the data is structured at the word level (one word + any punctuation per row), but our data is not. I have made some initial changes that should work. The output from this will be at the word level.

`daseg_example_output.xlsx` contains a small example of what daseg output should generally look like (all the cols, etc, may not match. We just want to have a word per row, to start.)

`DasegZeiglerPoster 5.1.25.pdf` contains a poster presentation from my previous research, detailing some of the things we did! Check it out if you are interested.


### Graphing

In the `plotting` folder, there contains several scripts for some initial plots of the data.

`get_hists.py` will grab histograms for all files, and for the entire supplied directory. Useful for seeing what is happening in a directory.

`comparative_plotting.py` will compare two different directories across their frequencies.

`graph_file_da.py` is in-progress, with the purpose of exploring DAs throughout a transcript, rather than just at the frequency level.


## Setup

Use requirements.txt to download required packages. I did this on Python 3.13.3, so you may need to be on a similar version for some imports to work.

## Resources

Labeling manual for dialogue acts: https://web.stanford.edu/~jurafsky/ws97/manual.august1.html

Zelasko paper: https://arxiv.org/pdf/2107.02294



