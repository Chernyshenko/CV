import os
import gc
import sys
import json
import glob
import numpy as np
from argparse import ArgumentParser
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset import OCRDataset
from transform import Resize#, Rotate, Pad
from common import collate_fn_ocr, abc
from model import CRNN

def eval(net, data_loader, device):
	count, tp, avg_ed = 0, 0, 0
	iterator = tqdm.tqdm(data_loader)

	with torch.no_grad():
		for batch in iterator:
			images = batch['images'].to(device)
			out = net(images, decode=True)
			gt = (batch['seqs'].numpy() - 1).tolist()
			lens = batch['seq_lens'].numpy().tolist()

			pos, key = 0, ''
			for i in range(len(out)):
				gts = ''.join(abc[c] for c in gt[pos:pos + lens[i]])
				pos += lens[i]
				if gts == out[i]:
					tp += 1
				else:
					avg_ed += editdistance.eval(out[i], gts)
				count += 1

	acc = tp / count
	avg_ed = avg_ed / count

	return acc, avg_ed

def train(model,criterion, optimizer, scheduler, train_dataloader, val_dataloader, path, num_epoch=1, device=None):

	best_acc_val = -1
	for epoch in range(num_epoch):
		#logger.info('Starting epoch {}/{}.'.format(epoch + 1, num_epoch))
		model.train()
		epoch_losses = []
		print_loss = []

		for i, batch in enumerate(tqdm.tqdm(train_dataloader, total=len(val_dataloader), leave=False, position=0)):
			images = batch["image"].to(device)
			seqs_gt = batch["seq"]
			seq_lens_gt = batch["seq_len"]

			seqs_pred = model(images).cpu()
			log_probs = F.log_softmax(seqs_pred, dim=2)
			seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

			loss = criterion(log_probs, seqs_gt, seq_lens_pred, seq_lens_gt)  

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print_loss.append(loss.item())
			if (i + 1) % 100 == 0:
				mean_loss = np.mean(print_loss)
				print(f'Loss: {mean_loss:.7f}')
				scheduler.step(mean_loss)
				print_loss = [] 

			epoch_losses.append(loss.item())

		#logger.info('Epoch finished! Loss: {:.5f}'.format(np.mean(loss_mean)))
		print(i, np.mean(epoch_losses))

		model.eval()
		'''
		acc_val, acc_ed_val = eval(model, val_dataloader, device=device)

		if acc_val > best_acc_val:
			best_acc_val = acc_val
			#torch.save(model.state_dict(), path)
			#logger.info('Valid acc: {:.5f}, acc_ed: {:.5f} (best)'.format(acc_val, acc_ed_val))
			print('Valid acc: {:.5f}, acc_ed: {:.5f} (best)'.format(acc_val, acc_ed_val))
		else:
			print('Valid acc: {:.5f}, acc_ed: {:.5f} (best {:.5f})'.format(acc_val, acc_ed_val, best_acc_val))
			#logger.info('Valid acc: {:.5f}, acc_ed: {:.5f} (best {:.5f})'.format(acc_val, acc_ed_val, best_acc_val))
		'''

	#logger.info('Best valid acc: {:.5f}'.format(best_acc_val))
	torch.save(model.state_dict(), path)

def load_json(file):
	with open(file, 'r') as f:
		return json.load(f)


def main():

	parser = ArgumentParser()
	parser.add_argument('-d', '--data_path', dest='data_path', type=str, default= '../../data/',help='path to the data')
	parser.add_argument('--epochs', '-e', dest='epochs', type=int, help='number of train epochs', default=2)
	parser.add_argument('--batch_size', '-b', dest='batch_size', type=int, help='batch size', default=16) 
	parser.add_argument('--load', '-l', dest='load', type=str, help='pretrained weights', default=None)
	parser.add_argument('-v', '--val_split', dest='val_split', default=0.8, type=float, help='train/val split')
	parser.add_argument('--augs', '-a', dest='augs', type=float, help='degree of geometric augs', default=0)

	args = parser.parse_args()
	OCR_MODEL_PATH = '../pretrained/ocr.pt'

	all_marks = load_json(os.path.join(args.data_path, 'train.json'))
	test_start = int(args.val_split * len(all_marks))
	train_marks = all_marks[:test_start]
	val_marks = all_marks[test_start:]

	w, h = (320, 64)
	train_transforms = transforms.Compose([
		#Rotate(max_angle=args.augs * 7.5, p=0.5),  # 5 -> 7.5
		#Pad(max_size=args.augs / 10, p=0.1),
		Resize(size=(w, h)),
		transforms.ToTensor()
	])
	val_transforms = transforms.Compose([
		Resize(size=(w, h)),
		transforms.ToTensor()
	])
	alphabet = abc

	train_dataset = OCRDataset(marks=train_marks, img_folder=args.data_path, alphabet=alphabet, transforms=train_transforms)
	val_dataset = OCRDataset(marks=val_marks, img_folder=args.data_path, alphabet=alphabet, transforms=val_transforms)

	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=0, 
	    collate_fn=collate_fn_ocr, timeout=0, shuffle=True)

	val_dataloader = DataLoader( val_dataset, batch_size=args.batch_size, drop_last=False, num_workers=0,
	    collate_fn=collate_fn_ocr, timeout=0)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	model = CRNN(alphabet)
	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, amsgrad=True, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
	criterion = F.ctc_loss

	try:
		train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, OCR_MODEL_PATH, args.epochs, device)
	except KeyboardInterrupt:
		torch.save(model.state_dict(),  OCR_MODEL_PATH + 'INTERRUPTED_')
		#logger.info('Saved interrupt')
		sys.exit(0)

if __name__ == '__main__':
	sys.exit(main())
