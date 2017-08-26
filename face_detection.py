import os
import dlib
import scipy.misc
import numpy as np 

face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

tolerance = 0.6

# This function will take an image and return its face encodings using the neural network
def get_face_encodings(path_to_image):
	image = scipy.misc.imread(path_to_image)

	detected_faces = face_detector(image, 1)

	shapes_faces = [shape_predictor(image, face) for face in detected_faces]

	return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) 
							for face_pose in shapes_faces]

# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
	return (np.linalg.norm(known_faces - face, axis=1) <= tolerance)

# This function returns the name of the person whose image matches with the given face (or 'Not Found')
# known_faces is a list of face encodings
# names is a list of the names of people (in the same order as the face encodings - to match the name with an encoding)
# face is the face we are looking for
def find_match(known_faces, names, face):
	matches = compare_face_encodings(known_faces, face)
	
	count = 0
	for match in matches:
		if match:
			return names[count]
		count += 1
	
	return 'Unknown'


image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('images/'))

image_filenames = sorted(image_filenames)

paths_to_images = ['images/' + x for x in image_filenames]

face_encodings = []

for path_to_image in paths_to_images:
	face_encodings_in_image = get_face_encodings(path_to_image)

	if len(face_encodings_in_image) != 1:
		print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
		exit()

	face_encodings.append(face_encodings_in_image[0])

test_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('test/'))

paths_to_test_images = ['test/' + x for x in test_filenames]

names = [x[:-4] for x in image_filenames]

for path_to_image in paths_to_test_images:
	face_encodings_in_image = get_face_encodings(path_to_image)
	length = len(face_encodings_in_image)
	print(length)

	x = 0
	if length > 1:
		while (length > 1) & (x < length):
			image = face_encodings_in_image[x]
			match = find_match(face_encodings, names, image)
			print(path_to_image, match)
			x = x + 1
	else:
		match = find_match(face_encodings, names, face_encodings_in_image[0])
		print(path_to_image, match)