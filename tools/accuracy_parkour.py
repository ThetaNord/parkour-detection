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

import os, sys
import argparse
import cv2
import logging

import numpy as np

from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import load_object

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

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
        '--thresh',
        dest='thresh',
        help='detection prob threshold',
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--box-idx',
        dest='box_idx',
        help='the id of the class that contains the relevant annotations',
        default=1,
        type=int
    )
    parser.add_argument(
        '--verbose',
        dest='long_output',
        help='toggle long-form output of statistics',
        action='store_true'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

# In-depth stats calculation
def complete_stats(dataset, detections_pkl, threshold, box_idx, long_output=True):
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

	true_positives = 0
	true_negatives = 0
	false_positives = 0
	false_negatives = 0
	total = len(roidb)
	# Iterate through all images
	for ix, entry in enumerate(roidb):
		cls_boxes_i = [
			id_or_index(ix, cls_k_boxes) for cls_k_boxes in all_boxes
		]
		preds = np.array(cls_boxes_i[box_idx])
		true_boxes = entry['boxes']
		# Check if the images resulted in a true/false positive/negative
		if (true_boxes.shape[0] == 0):
			if preds.shape[0] > 0 and np.max(preds[:,4]) > threshold:
				false_positives += 1
			else:
				true_negatives += 1
		else:
			if preds.shape[0] > 0 and np.max(preds[:,4]) > threshold:
				true_positives += 1			
			else:
				false_negatives += 1

	# Calculate the statistics
	prec = float('nan')
	if true_positives+false_positives > 0:
		prec = true_positives/float(true_positives+false_positives)
	elif false_negatives == 0:
		prec = 1.
	rec = float('nan')
	if true_positives+false_negatives > 0:
		rec = true_positives/float(true_positives+false_negatives)
	elif false_positives == 0:
		rec = 1.
	acc = float(true_positives+true_negatives)/total
	fm = 0
	if prec > 0 or rec > 0:
		fm = 2.0*prec*rec/(prec+rec)
	# Re-enable printing
	enablePrint()
	# Print results
	if (long_output):
		print("True positives: {}\tFalse positives: {}".format(true_positives, false_positives))
		print("True negatives: {}\tFalse negatives: {}".format(true_negatives, false_negatives))
		print("Total: {}".format(total))
		print("Precision: " + str(prec*100))
		print("Recall: " + str(rec*100))
		print("F-measure: " + str(fm*100))
		print("Accuracy: " + str(acc*100))
	else:
		print("{};{};{};{};".format(acc, prec, rec, fm))
	return acc

# Accuracy calculation
def accuracy(dataset, detections_pkl):
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

	trues = 0.
	# Iterate through all images
	for ix, entry in enumerate(roidb):
		cls_boxes_i = [
			id_or_index(ix, cls_k_boxes) for cls_k_boxes in all_boxes
		]
		true_boxes = entry['boxes']
		if (true_boxes.shape[0] == 0) == (len(cls_boxes_i[3]) == 0):
			trues += 1
	# Finally, calculate accuracy by dividing the sum of true predictions by total samples
	acc = trues/len(roidb)
	print("Accuracy: " + str(acc))
	return acc

if __name__ == '__main__':
	# "Block" printing to stdout to prevent lines that are not stats
	blockPrint()
	# Parse command line arguments
	opts = parse_args()
	complete_stats(
		opts.dataset,
		opts.detections,
		opts.thresh,
		opts.box_idx,
		opts.long_output
	)
