import pandas as pd
import argparse

from tqdm import tqdm
from matplotlib import pyplot as plt

#import auxiliary functions for python
import sys

from joblib import dump, load

sys.path.append('/Users/kipnisal/')
from AuthAttLib.AuthAttLib import *
from tqdm import *
from HCsimCls import *
import logging
logging.basicConfig(level=logging.INFO)

# data_file = '/scratch/users/kipnisal/Data/PAN/pan20-authorship-verification-training-small-truth.jsonl'
# ground_truth_file = '/scratch/users/kipnisal/Data/PAN/pan20-authorship-verification-training-small-truth.jsonl'

NG_RANGE = (1, 1)
VOCAB_SIZE = 500
VOCAB_FILE = f'./most_freq_NG{NG_RANGE}.csv'


def test_model(md, df_test) :

	y = df_test.same
	X = df_test.texts

	HC_vals = []
	probs = []
	logging.info(f"Predicting on training set...")
	for texts in tqdm(X) :
		HC_vals.append(md.get_HCsim(texts[0], texts[1]))
		probs.append(md.predict_proba(texts[0], texts[1]))

	res = pd.DataFrame(y)
	res['prob'] = probs
	res['HC'] = HC_vals
	
	valids = np.array(HC_vals_hat) != .5
	recall = np.mean(valids)

	acc = np.mean((np.array(HC_vals_hat)[valids] > .5) == y[valids])
	logging.info(f"Training accuracy = {acc}, recall = {recall}")
	return res


def main():
	parser = argparse.ArgumentParser()
	parser = argparse.ArgumentParser(description='PAN-20 Cross-domain Authorship Verification task: HC similarity')
	parser.add_argument('-i', type=str, help='training data')
	parser.add_argument('-l', type=str, help='labels')
	parser.add_argument('-m', type=str, help='model file', default='model_trained.joblib')
	parser.add_argument('-o', type=str, help='results file', default='results.csv')
	parser.add_argument('-f', type=float, help='fraction', default=1.0)
	args = parser.parse_args()
	if not args.i:
		print('ERROR: missing data file')
		parser.exit(1)
	
	# read data:
	logging.info(f"Reading test data from {args.i}...")
	data_file = args.i
	ground_truth_file = args.l

	df = pd.read_json(data_file, lines=True)       
	dft = pd.read_json(ground_truth_file, lines=True)
	assert(len(df) == len(dft)) 
	logging.info(f"Found {len(df)} author-pairs")

	df_all = df.merge(dft, on = 'id', how = 'left')
	df_test = df_all.sample(frac = args.f)

	df_test = df_test.rename(columns = {'pair' : 'texts'})
	md = load(args.m)
	res = test_model(md, df_test)
	res.to_csv(args.o)
	

if __name__ == '__main__':
	main()

