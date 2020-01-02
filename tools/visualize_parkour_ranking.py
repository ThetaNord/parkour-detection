# Copyright (c) 2019, Tuure Saloheimo, Aalto University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import os, sys, json
import argparse
import cv2

import numpy as np

from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import load_object

from geopy.distance import distance

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import ImageGrid

# Color values for prediction and ground-truth boxes
_PRED_COLOR = (255, 0, 255)
_GT_COLOR = (0, 255, 0)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dataset',
		dest='dataset',
		help='dataset',
		default='coco_2014_minival',
		type=str
	)
	parser.add_argument(
		'--detections',
		dest='detections',
		help='detections pkl file',
		default='',
		type=str
	)
	parser.add_argument(
		'--coordinate_file',
		dest='coord_file',
		help='json file with locations for the images',
		default='',
		type=str
	)
	parser.add_argument(
		'--min-distance',
		dest='min_distance',
		help='minimum distance between predicted spots (in meters)',
		default=0.0,
		type=float
	)
	parser.add_argument(
		'--thresh',
		dest='threshold',
		help='detection prob threshold',
		default=0.59,
		type=float
	)
	parser.add_argument(
		'--box-line',
		dest='box_thickness',
		help='line thickness for bounding boxes',
		default=3,
		type=int
	)
	parser.add_argument(
		'--im-count',
		dest='image_count',
		help='number of samples, both positive and negative',
		default=10,
		type=int
	)
	parser.add_argument(
		'--box-count',
		dest='box_count',
		help='number of boxes to account for in scoring',
		default=2,
		type=int
	)
	parser.add_argument(
		'--class-id',
		dest='class_id',
		help='id of the class to visualize',
		default=1,
		type=int
	)
	parser.add_argument(
		'--angle',
		dest='angle',
		help='whether the ranking should be shown in rows (hor) or columns (ver)',
		default='hor',
		type=str
	)
	parser.add_argument(
		'--seed',
		dest='seed',
		help="the seed to use for initializing numpy's random number generator",
		default=0,
		type=int
	)
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	args = parser.parse_args()
	return args

def show_imgspec(gspec, im, row, col):
	ax = plt.subplot(gs[i,j])
	ax.imshow(im)
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])

def show_img(im_grid, im, grid_idx):
	im_grid[grid_idx].imshow(im)
	im_grid[grid_idx].grid(False)
	im_grid[grid_idx].set_xticks([])
	im_grid[grid_idx].set_yticks([])

def vis_bbox(img, bbox, color, thickness=1):
	# Visualizes a bounding box
	img = img.astype(np.uint8)
	(x0, y0, w, h) = bbox
	x1, y1 = int(x0 + w), int(y0 + h)
	x0, y0 = int(x0), int(y0)
	cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
	return img

def add_negative(current_entries, new_entry, coord_data, min_distance):
	new_entry['coords'] = coord_data[os.path.split(new_entry['image'])[-1]]
	store = True
	for entry in current_entries:
		if distance((new_entry['coords'][0], new_entry['coords'][1]), (entry['coords'][0], entry['coords'][1])).km*1000 < min_distance:
			store = False
			break
	if store:
		current_entries.append(new_entry)
	return current_entries

def visualize_ranking(dataset, detections_pkl, opts):

	# Load predictions and ground truths
	ds = JsonDataset(dataset)
	roidb = ds.get_roidb(gt=True)
	dets = load_object(detections_pkl)
	all_boxes = dets['all_boxes']

	def id_or_index(ix, val):
		if len(val) == 0:
			return val
		else:
			return val[ix]

	# Load coordinates
	with open(opts.coord_file) as json_file:
		coord_data = json.load(json_file)

	# Iterate through all images and note false positive and negatives, as well as entry scores
	false_positives = []
	false_negatives = []
	scores = []
	for ix, entry in enumerate(roidb):
		cls_boxes_i = [
			id_or_index(ix, cls_k_boxes) for cls_k_boxes in all_boxes
		]
		preds = np.array(cls_boxes_i[opts.class_id])
		entry['preds'] = preds
		true_boxes = entry['boxes']
		if preds.shape[0] > 0 and np.max(preds[:,-1]) > opts.threshold:
			box_scores = preds[:,-1]
			box_scores = box_scores[np.where(box_scores > opts.threshold)]
			score = np.sum(box_scores[np.argsort(box_scores)[-opts.box_count:]])
			scores.append([entry, score])
			if true_boxes.shape[0] == 0:
				false_positives.append(entry)
		else:
			if true_boxes.shape[0] > 0:
				false_negatives = add_negative(false_negatives, entry, coord_data, opts.min_distance)


	# Find top rated entries
	scores = np.array(scores)
	scores = scores[np.argsort(scores[:,1])[::-1]]
	
	for entry in scores[:,0]:
		entry['coords'] = coord_data[os.path.split(entry['image'])[-1]]
	# Filter by proximity
	for i in range(scores.shape[0]):
		if scores[i][1] > 0:
			current_entry = scores[i][0]
			for j in range(i+1, scores.shape[0]):
				second_entry = scores[j][0]
				dist = distance((current_entry['coords'][0], current_entry['coords'][1]), (second_entry['coords'][0], second_entry['coords'][1])).km*1000
				if dist < opts.min_distance:
					scores[j][1] = 0
	scores = scores[np.where(scores[:,1] > 0)]
	top_entries = scores[np.argsort(scores[:,1])[-opts.image_count:][::-1]]

	# Choose random negative samples
	false_samples = np.append(false_negatives, false_positives)
	np.random.shuffle(false_samples)

	# Visualize positive and negative samples
	rows_cols = (opts.image_count, 2) if opts.angle == 'ver' else (2, opts.image_count)
	plt_shape = (6., opts.image_count*2.5) if opts.angle == 'ver' else (opts.image_count*2.5, 6.)
	fig = plt.figure(1, plt_shape)

	grid = ImageGrid(fig, 111,
				nrows_ncols=rows_cols,
				axes_pad=0.03,
				label_mode='L',
			)
	# Show top ranked images
	for i, result in enumerate(top_entries):
		entry = result[0]
		score = result[1]
		grid_idx = i
		if opts.angle == 'ver':
			grid_idx = i*2
		# Load image and add bounding boxes
		im = cv2.imread(entry['image'])
		preds = entry['preds']
		true_boxes = entry['boxes']
		for bbox in true_boxes:
			im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), _GT_COLOR, opts.box_thickness)
		count = 0
		for bbox in preds:
			if bbox[-1] > opts.threshold:
				count += 1
				print(os.path.split(entry['image'])[-1] + ': ' + str(bbox[0:4]))
				im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), _PRED_COLOR, opts.box_thickness)
			if count >= opts.box_count:
				break
		# Adjust grid setting
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		show_img(grid, im, grid_idx)
		t = grid[grid_idx].text(12, 42, "Score: " + str(round(score, 3)), fontsize=8, bbox=dict(boxstyle='square', fc='white', ec='none', alpha=0.6))
		if i == 0:
			if opts.angle == 'ver':
				grid[grid_idx].set_title("Top\nPredictions", size=18)
			else:
				grid[grid_idx].set_ylabel("Top Predictions", fontsize=13)
	# Show random negative samples (false positive, false negative)
	for i, entry in enumerate(false_samples):
		if i >= opts.image_count:
			break
		grid_idx = opts.image_count+i
		if opts.angle == 'ver':
			grid_idx = 2*i+1
		# Load image and add bounding boxes
		im = cv2.imread(entry['image'])
		preds = entry['preds']
		true_boxes = entry['boxes']
		for bbox in true_boxes:
			im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), _GT_COLOR, opts.box_thickness)
		for bbox in preds:
			if bbox[-1] > opts.threshold:
				im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), _PRED_COLOR, opts.box_thickness)
		# Adjust grid setting
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		grid[grid_idx].imshow(im)
		grid[grid_idx].grid(False)
		grid[grid_idx].set_xticks([])
		grid[grid_idx].set_yticks([])
		if i == 0:
			if opts.angle == 'ver':
				grid[grid_idx].set_title("Errors", size=18)
			else:
				grid[grid_idx].set_ylabel("Errors", fontsize=13)
	plt.axis('off')
	plt.subplots_adjust(hspace=1)
	plt.savefig("ranking.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
	opts = parse_args()
	np.random.seed(opts.seed)
	visualize_ranking(
		opts.dataset,
		opts.detections,
		opts
	)

