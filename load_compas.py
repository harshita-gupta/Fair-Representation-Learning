import os
import pandas as pd


filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'compas-scores-two-years.csv')

try:
    df = pd.read_csv(filepath, index_col='id')

except IOError as err:
    print("IOError: {}".format(err))
    print("To use this class, please download the following file:")
    print("\n\thttps://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
    print("\nand place it, as-is, in the folder:")
    print("\n\t{}\n".format(os.path.abspath(os.path.join(
       os.path.abspath(__file__), '..', '..', 'data', 'raw', 'compas'))))
    import sys
    sys.exit(1)
