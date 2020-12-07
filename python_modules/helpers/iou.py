def compute_iou(boxA, boxB):
	"""
	input: Bounding bixes to calculate the iou for
			Box notation: [x1, y1, x2, y2], with:
			 ______  y1
			|      |
			|      |
			|      |
			|______| y2
			x1     x2
	
	Definition of the intersection rectangle:
	 ______
	|      |
	|    __|___  yA
	|   |  |   |
	|___|__|   | yB
        |      |
		|______|
	   xA xB
 
	"""

	# proof if the box coordinates are valid 
	if((boxA[0] >= boxA[2]) or (boxA[1] > boxA[3])):
		print("BoxA is not valid: {}".format(boxA))
		return 0.0
	if((boxB[0] >= boxB[2]) or (boxB[1] > boxB[3])):
		print("BoxB is not valid: {}".format(boxB))
		return 0.0

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = (xB - xA) * (yB - yA)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the intersection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
    
	# return the intersection over union value
	return iou