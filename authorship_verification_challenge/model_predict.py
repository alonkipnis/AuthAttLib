#!/usr/bin/python3
from __future__ import print_function
import os
import json
import argparse
from joblib import dump, load
import sys
import numpy as np

# training == creating an instance of HCsimCls with the correct vocabulary 

# Applies the model to evaluation data
# Produces an output file predictions.jsonl

#from HCsimSolver import HCsimSolver

def apply_model(eval_folder, output_file, model_file):
    model = load(model_file) 
    answers=[]
    with open(output_file, 'w') as outfile:
        with open(eval_folder+os.sep+'pairs.jsonl', 'r') as fp:
            for i,line in enumerate(fp):
                try :
                    X=json.loads(line)
                    hc = model.get_HCsim(X['pair'][0],X['pair'][1])
                    pred = model.predict_proba(X['pair'][0],X['pair'][1])
                    print(i+1,X['id'],round(pred,3))
                    
                    json.dump({'id': X['id'],'value': round(pred,3), 'HC' : hc}, outfile)
                    outfile.write('\n')
                except :
                    print("could not evaluate pair ", X['id'])
                    json.dump({'id': X['id'],'value': 0.5, 'HC' : 'NA'}, outfile)
                    outfile.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='PAN-20 Cross-domain Authorship Verification task: HC similarity')
    parser.add_argument('-i', type=str, help='Evaluation directory')
    parser.add_argument('-o', type=str, help='Output file')
    parser.add_argument('-m', type=str, help='model file')
    args = parser.parse_args()
    if not args.i:
        print('ERROR: The input file is required')
        parser.exit(1)
    if not args.o:
        print('ERROR: The output file is required')
        parser.exit(1)
    
    apply_model(args.i, args.o, args.m)

if __name__ == '__main__':
    main()