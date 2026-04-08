import pandas as pd
from pathlib import Path
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Token classification with Longformer pipeline.")

    parser.add_argument('--directory', type=str, help="Directory containing CSV, TSV, or XLSX files")
    parser.add_argument('--col1', type=str)
    parser.add_argument('--col2', type=str)


    args = parser.parse_args()
    dir_path = Path(args.directory)
    col1 = args.col1
    col2 = args.col2

    for file in dir_path.iterdir():
        if file.suffix.lower() in {'.csv', '.tsv', '.xlsx'}:
            if file.suffix == '.tsv':
                df = pd.read_csv(file, delimiter='\t')
            elif file.suffix == '.xlsx':
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)

            last_row = df.iloc[-1]
            match = last_row[col1] == last_row[col2]
            print(f"{file.name}: {'match' if match else 'mismatch'}")