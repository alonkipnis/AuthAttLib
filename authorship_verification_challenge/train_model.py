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


def fit_model(vocab, df_train) :

	md = HCsimCls(vocab=vocab, ng_range=NG_RANGE, gamma=0.2, radius=0.0001)
	y = df_train.same
	X = df_train.texts

	HC_vals = []
	for texts in tqdm(X) :
		HC_vals.append(md.get_HCsim(texts[0], texts[1]))

	vals0 = np.array(HC_vals)[np.array(y) == 1]  #same
	vals1 = np.array(HC_vals)[np.array(y) == 0]	 #not same

	mu0, std0 = np.mean(vals0), np.std(vals0)
	mu1, std1 = np.mean(vals1), np.std(vals1)
	logging.info(f"Fitting completed. N0 = N({mu0}, {std0}^2), N1 = N({mu1}, {std1}^2)")
	md.set_prob(mu0 = mu0, mu1 = mu1, sigma0 = std0, sigma1 = std1)

	logging.info(f"Predicting on training set...")
	HC_vals_hat = []
	for texts in tqdm(X) :
		r = md.predict_proba(texts[0],texts[1])
		HC_vals_hat.append(r)

	valids = np.array(HC_vals_hat) != .5
	recall = np.mean(valids)

	acc = np.mean((np.array(HC_vals_hat)[valids] > .5) == y[valids])
	logging.info(f"Training accuracy = {acc}, recall = {recall}")
	return md

def main():
	parser = argparse.ArgumentParser()
	parser = argparse.ArgumentParser(description='PAN-20 Cross-domain Authorship Verification task: HC similarity')
	parser.add_argument('-i', type=str, help='training data')
	parser.add_argument('-l', type=str, help='labels')
	parser.add_argument('-o', type=str, help='output model file', default='model_trained.joblib')
	parser.add_argument('-f', type=float, help='fraction', default=1.0)
	args = parser.parse_args()
	if not args.i:
		print('ERROR: missing data file')
		parser.exit(1)
	
	# read data:
	logging.info(f"Reading training data from {args.i}...")
	data_file = args.i
	ground_truth_file = args.l

	df = pd.read_json(data_file, lines=True)       
	dft = pd.read_json(ground_truth_file, lines=True)
	assert(len(df) == len(dft)) 
	logging.info(f"Found {len(df)} author-pair")

	df_all = df.merge(dft, on = 'id', how = 'left')
	df_train = df_all.sample(frac = args.f)

	# read vocabulary:
	#logging.info(f"Reading vocab data from {VOCAB_FILE}.")
	most_freq = pd.read_csv(VOCAB_FILE).token.tolist()
	vocab = most_freq[:VOCAB_SIZE]
	#df_train['texts'] = df_train.pair.apply(eval)
	df_train = df_train.rename(columns = {'pair' : 'texts'})
	md = fit_model(vocab, df_train)

	dump(md, args.o)

if __name__ == '__main__':
	main()

