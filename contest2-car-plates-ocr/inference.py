import os, gc, sys, cv2, numpy as np
import json
import glob
import torch, torchvision
from argparse import ArgumentParser
import tqdm
from PIL import Image
from torchvision import transforms
import pandas as pd

from detection.rcnn import get_detector_model
from recognition.common import get_vocab_from_marks, four_point_transform, abc, decode
from recognition.model import CRNN
from recognition.transform import Resize

def simplify_contour(contour, n_corners=4):
	n_iter, max_iter = 0, 1000
	lb, ub = 0., 1.

	while True:
		n_iter += 1
		if n_iter > max_iter:
			print('simplify_contour didnt coverege')
			return None

		k = (lb + ub)/2.
		eps = k*cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, eps, True)

		if len(approx) > n_corners:
			lb = (lb + ub)/2.
		elif len(approx) < n_corners:
			ub = (lb + ub)/2.
		else:
			return approx
		
def load_json(file):
	with open(file, 'r') as f:
		return json.load(f)

class npEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.int32):
			return int(obj)
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

def main():
	parser = ArgumentParser()
	parser.add_argument('-d', '--data_path', dest='data_path', type=str, default='../data/', help='path to the data')
	parser.add_argument('-t', '--threshold', dest='threshold', type=float, default=0.93,
	                    help='decision threshold for segmentation model')
	parser.add_argument('-s', '--seg_model', dest='detector_model', type=str, default='pretrained/detector.pt',
	                    help='path to a trained detector model')
	parser.add_argument('-r', '--rec_model', dest='rec_model', type=str, default= 'pretrained/ocr.pt',
	                    help='path to a trained recognition model')
	args = parser.parse_args()

	print('Start inference')

	alphabet = abc

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	my_transforms = transforms.Compose([
		transforms.ToTensor()
	])

	model = get_detector_model()
	model.load_state_dict(torch.load(args.detector_model))
	model.to(device)
	model.eval()

	test_images = glob.glob(os.path.join(args.data_path, 'test\\*'))

	THRESHOLD_MASK = 0.05

	preds = []

	for file in tqdm.tqdm(test_images, position=0, leave=False):

		img = Image.open(file).convert('RGB')
		img_tensor = my_transforms(img)
		with torch.no_grad():
			predictions = model([img_tensor.to(device)])
		prediction = predictions[0]

		pred = dict()
		pred['file'] = file
		pred['nums'] = []

		for i in range(len(prediction['boxes'])):
			x_min, y_min, x_max, y_max = map(int, prediction['boxes'][i].tolist())
			label = int(prediction['labels'][i].cpu())
			score = float(prediction['scores'][i].cpu())
			mask = prediction['masks'][i][0, :, :].cpu().numpy()

			if score > args.threshold:            
				contours,_ = cv2.findContours((mask > THRESHOLD_MASK).astype(np.uint8), 1, 1)
				approx = simplify_contour(contours[0], n_corners=4)
				
				if approx is None:
					x0, y0 = x_min, y_min
					x1, y1 = x_max, y_min
					x2, y2 = x_min, y_max
					x3, y3 = x_max, y_max
				else:
					x0, y0 = approx[0][0][0], approx[0][0][1]
					x1, y1 = approx[1][0][0], approx[1][0][1]
					x2, y2 = approx[2][0][0], approx[2][0][1]
					x3, y3 = approx[3][0][0], approx[3][0][1]
					
				points = [[x0, y0], [x2, y2], [x1, y1],[x3, y3]]

				pred['nums'].append({
					'box': points,
					'bbox': [x_min, y_min, x_max, y_max],
				})

		preds.append(pred)   

		
	with open(os.path.join(args.data_path, 'test.json'), 'w') as json_file:
		json.dump(preds, json_file, cls=npEncoder)


	gc.collect()

	crnn = CRNN(abc)
	crnn.load_state_dict(torch.load(args.rec_model))
	crnn.to(device)
	crnn.eval()

	test_marks = load_json(os.path.join(args.data_path, 'test.json'))

	resizer = Resize()

	file_name_result = [] 
	plates_string_result = []

	for item in tqdm.tqdm(test_marks, leave=False, position=0):

	    img_path = item["file"]
	    img = cv2.imread(img_path)

	    results_to_sort = []
	    for box in item['nums']:
	        x_min, y_min, x_max, y_max = box['bbox']
	        img_bbox = resizer(img[y_min:y_max, x_min:x_max])
	        img_bbox = my_transforms(img_bbox)
	        img_bbox = img_bbox.unsqueeze(0)


	        points = np.clip(np.array(box['box']), 0, None)
	        img_polygon = resizer(four_point_transform(img, points))
	        img_polygon = my_transforms(img_polygon)
	        img_polygon = img_polygon.unsqueeze(0)

	        preds_bbox = crnn(img_bbox.to(device)).cpu().detach()
	        preds_poly = crnn(img_polygon.to(device)).cpu().detach()

	        preds = preds_poly + preds_bbox
	        num_text = decode(preds, alphabet)[0]

	        results_to_sort.append((x_min, num_text))

	    results = sorted(results_to_sort, key=lambda x: x[0])
	    num_list = [x[1] for x in results]

	    plates_string = ' '.join(num_list)
	    file_name = img_path[img_path.find('test\\'):]

	    file_name_result.append(file_name)
	    plates_string_result.append(plates_string)
	    
	df_submit = pd.DataFrame({'file_name': file_name_result, 'plates_string': plates_string_result})
	df_submit.to_csv('my_submission.csv', index=False)
	print('Done')


if __name__ == '__main__':
	main()    