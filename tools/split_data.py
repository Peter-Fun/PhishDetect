import os
import argparse
import pandas as pd
import random

'''
Takes the email dataset, adds a column for split assignment, and does the assignment.
'''

SPLIT_OPTIONS = ['train', 'test']
SPLIT_WEIGHTS = [0.8, 0.2]
accumulators = [0, 0]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Takes the email dataset and splits it.')
    parser.add_argument('csv_file', type=str, help='the input data csv file')
    parser.add_argument('out_file', type=str, help='the output data csv file')
    args = parser.parse_args()

    csv_file = args.csv_file
    out_file = args.out_file
    assert os.path.isfile(csv_file)

    with open(csv_file, 'r') as file:
        df = pd.read_csv(file)

    split_assignment = []
    for i in range(len(df)):
        assignment = random.choices(SPLIT_OPTIONS, SPLIT_WEIGHTS)[0]
        split_assignment.append(assignment)
        accumulators[SPLIT_OPTIONS.index(assignment)] += 1
    
    df['split'] = split_assignment

    # Summary Printing
    accumulators = [ str(i) for i in accumulators]
    print(f'Summary: {"/".join(SPLIT_OPTIONS)}={"/".join(accumulators)}')

    with open(out_file, 'w') as file:
        df.to_csv(file)

    
    