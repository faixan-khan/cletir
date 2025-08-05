import os, pickle, json, random, math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from utils import compute_map, ImageNetQueryset

from tqdm import tqdm
from agg_matcher import MatchATN

import config
from templates import imagenet_classes

import argparse

def get_args_parser():
	parser = argparse.ArgumentParser('CLIP retrieval', add_help=False)
	parser.add_argument('--model', type=str, choices=['clip', 'eva', 'meta','open', 'sig'], default='clip',help='CLip version')
	parser.add_argument('--v_model', type=str, choices=['deit','dino'], default='dino',help='CLip version')
	parser.add_argument('--root_dir', default='./queries/', type=str)
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--method', default='', type=str, choices=['', '_classdbr', '_dbr'])
	parser.add_argument('--model_path', default='model/model.pth', type=str)
	return parser

device = "cuda" if torch.cuda.is_available() else "cpu"

res = {}

def read_file(path):
	if path.endswith('.pkl'):
		with open(path,'rb') as f:
			return pickle.load(f)
	else:
		return np.load(path)

class TestDataset(Dataset):
	def __init__(self, root_dir, preprocess=None, dataset=None, model_name=None, args=None):
		self.root_dir = root_dir
		self.dataset = dataset
		self.model_name = model_name
		self.v_model = args.v_model
		self.method = args.method


		self.q_img = read_file(os.path.join(self.root_dir, 'aggregator', self.v_model, self.dataset + '_{}_q_img{}.pkl'.format(self.v_model, self.method)))
		self.q_text = read_file(os.path.join(self.root_dir, 'aggregator', self.model_name, self.dataset + '_clip_q_text{}.pkl'.format(self.method)))

		# self.q_img = read_file(os.path.join(self.root_dir, 'aggregator', self.v_model, self.dataset + '_{}_q_img{}.pkl'.format(self.v_model, self.method)))

		# q_text_file = os.path.join(self.root_dir, 'aggregator', self.model_name, self.dataset + '_clip_q_text{}.pkl'.format(self.method))
		# self.q_text = read_file(q_text_file)
		
		if 'imagenet' in self.dataset:
			self.test_classes = imagenet_classes
		else:
			self.test_classes = list(self.q_img.keys())
		q_img = {key: value for key, value in self.q_img.items() if key in self.test_classes}
		q_text = {key: value for key, value in self.q_text.items() if key in self.test_classes}
		self.q_img = q_img
		self.q_text = q_text
	def __len__(self):
		return len(self.test_classes)
	def __getitem__(self, idx):
		label = self.test_classes[idx]
		img_feats = self.q_img[label][0]
		text_feats = self.q_text[label]
		return torch.from_numpy(img_feats).float().T, torch.from_numpy(text_feats).float(), label
	


def evaluate(model, test_dataset=None, model_name=None, args=None):

	clip_vecs = read_file(os.path.join(args.root_dir, 'aggregator', model_name, test_dataset + '_val_clip_db.npy'))
	dino_vecs = read_file(os.path.join(args.root_dir, 'aggregator', args.v_model, test_dataset + '_val_{}_db.npy'.format(args.v_model)))
	with open(os.path.join(args.root_dir, 'aggregator', test_dataset + '_val_cfg.json'), 'r') as json_file: 
		cfg = json.load(json_file)
	cfg = {int(key): value for key, value in cfg.items()}
	model.eval()
	dataloader = TestDataset(root_dir=args.root_dir, dataset=test_dataset, model_name=args.model, args=args)
	q_set = DataLoader(dataloader, batch_size=1, shuffle=False)

	agg_qvecs_img = []
	orig_qvecs_text = []
	avg_qvecs_img = []

	for idx,(img_feats, text_feats, label) in enumerate(q_set):
		if text_feats.shape[1] > 1 and len(text_feats.shape) > 2:
			text_feats = torch.mean(text_feats, dim=1)
		with torch.no_grad():
			q_vec, lamda = model(img_feats)
		lamda = lamda.cpu().data.squeeze().numpy()
		q_vec = q_vec.cpu().data.squeeze().numpy()
		text_feats = text_feats.cpu().data.squeeze().numpy()
		agg_qvecs_img.append(q_vec)
		orig_qvecs_text.append(text_feats)

	agg_qvecs_img = np.array(agg_qvecs_img)
	orig_qvecs_text = np.array(orig_qvecs_text)

	print('Dataset: ',test_dataset)

	scores = np.dot(clip_vecs, orig_qvecs_text.T)
	ranks = np.argsort(-scores, axis=0)
	map, aps, pr, prs, _, _ = compute_map(ranks, cfg)
	print('text: ', map)

	con = np.concatenate((orig_qvecs_text*(1-lamda), lamda*agg_qvecs_img), axis=1)
	con_qvecs = np.array(con)
	con_vecs = np.concatenate((clip_vecs, dino_vecs), axis=1)
	scores = np.dot(con_vecs, con_qvecs.T)
	ranks = np.argsort(-scores, axis=0)
	map, aps, pr, prs, _, _ = compute_map(ranks, cfg)
	print('lam: ', map)
	return map
	
datasets = ['imagenet', 'dtd', 'cars', 'sun', 'food', 'fgvc', 'pets', 'caltech', 'flowers', 'ucf', 'k700', 'r45', 'cifar10', 'cifar100', 'places']


def main(args, model):
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	model.to(device)	
	model = nn.DataParallel(model)
	model.load_state_dict(torch.load(args.model_path))
	model.to(device)
	model.eval()
	current_map = 0
	for d in datasets:
		current_map += evaluate(model, d, args.model, args)
	print('Avg map: ', current_map/len(datasets))

if __name__ == '__main__':
	args = get_args_parser()
	args = args.parse_args()
	print('Inference started', args)

	model = MatchATN(config.d_model, config.nhead, config.dropout, config.attn_layers)
	model = model.cuda()
	main(args, model)