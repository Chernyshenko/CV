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
from transform import Compose, Resize, Crop, Pad, Flip

from dataset import DetectionDataset
from rcnn import get_detector_model

def dice_coeff(input, target):
    smooth = 1.

    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum()
    union = input_flat.sum() + target_flat.sum()

    return (2. * intersection + smooth) / (union + smooth)


def eval_net(net, dataset, device):
    net.eval()
    tot = 0.
    with torch.no_grad():
        for i, b in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            imgs, true_masks = b
            masks_pred = net(imgs).squeeze(1)  # (b, 1, h, w) -> (b, h, w)
            masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
            tot += dice_coeff(masks_pred.cpu(), true_masks).item()
    return tot / len(dataset)

def train(model, optimizer, scheduler, train_dataloader, val_dataloader, path, num_epoch=1, device=None):

	model.train()

	for epoch in range(num_epoch):
		print_loss = []
		for i, (images, targets) in tqdm.tqdm(enumerate(train_dataloader), leave=False, position=0, total=len(train_dataloader)):

			images = [image.to(device) for image in images]
			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

			loss_dict = model(images, targets)
			losses = sum(loss_dict.values())

			losses.backward()
			optimizer.step()
			optimizer.zero_grad()
			
			print_loss.append(losses.item())
			if (i + 1) % 50 == 0:
				mean_loss = np.mean(print_loss)
				print(f'Loss: {mean_loss:.7f}')
				scheduler.step(mean_loss)
				print_loss = [] 


	torch.save(model.state_dict(), path)
	model.eval()
	#val_dice = eval_net(model, val_dataloader, device=device)
	#print(val_dice)
	#logger.info('Validation Dice Coeff: {:.5f} (best {:.5f})'.format(val_dice, best_model_info['val_dice']))
	

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():

	parser = ArgumentParser()
	parser.add_argument('-d', '--data_path', dest='data_path', type=str, default='../../data/' ,help='path to the data')
	parser.add_argument('-e', '--epochs', dest='epochs', default=1, type=int, help='number of epochs')
	parser.add_argument('-b', '--batch_size', dest='batch_size', default=1, type=int, help='batch size')
	parser.add_argument('-v', '--val_split', dest='val_split', default=0.8, type=float, help='train/val split')

	args = parser.parse_args()

	DETECTOR_MODEL_PATH = '../pretrained/detector.pt'

	all_marks = load_json(os.path.join(args.data_path, 'train.json'))
	test_start = int(args.val_split * len(all_marks))  
	train_marks = all_marks[:test_start]
	val_marks = all_marks[test_start:]

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	my_transforms = transforms.Compose([
		transforms.ToTensor()
	])

	train_dataset = DetectionDataset(marks=train_marks, img_folder=args.data_path, transforms=my_transforms)
	val_dataset = DetectionDataset(marks=val_marks, img_folder=args.data_path, transforms=my_transforms)

	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=4, collate_fn=collate_fn)
	val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, num_workers=4, collate_fn=collate_fn)

	torch.cuda.empty_cache()
	gc.collect()
	model = get_detector_model()

	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)

	try:
		train(model, optimizer, scheduler, train_dataloader, val_dataloader, DETECTOR_MODEL_PATH, args.epochs, device=device)
	except KeyboardInterrupt:
		torch.save(model.state_dict(),  DETECTOR_MODEL_PATH + '_INTERRUPTED')
		#logger.info('Saved interrupt')
		sys.exit(0)


if __name__ == '__main__':
	main()