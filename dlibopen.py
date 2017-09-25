import sys
import dlib
from skimage import io
import cv2
import os
import scipy.misc
import numpy as np 

webcam = cv2.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()

face_pose_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

cv2.namedWindow("Live Feed for Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Feed for Recognition", 600, 400)

tolerance = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX

def get_new_face_encodings(frame, detected_faces):
	shapes_faces = [shape_predictor(frame, face) for face in detected_faces]

	return [np.array(face_recognition_model.compute_face_descriptor(frame, face_pose, 1)) 
							for face_pose in shapes_faces]

def get_face_encodings(path_to_image):
	image = scipy.misc.imread(path_to_image)

	detected_faces = face_detector(image, 1)

	shapes_faces = [shape_predictor(image, face) for face in detected_faces]

	return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) 
							for face_pose in shapes_faces]

def draw_rectangle(top, left, right, bottom):
	cv2.rectangle(frame, (left, top), (right, bottom), (150, 150, 0), 4)
	cv2.rectangle(frame, (left-2, top-10), (right+2, top), (150, 151, 0), -1)

def compare_face_encodings(known_faces, face):
	return (np.linalg.norm(known_faces - face, axis=1) <= tolerance)

def find_match(known_faces, names, face):
	matches = compare_face_encodings(known_faces, face)
	
	count = 0
	for match in matches:
		if match:
			return names[count]
		count += 1
	
	return 'Unknown'

################################## Start of the program ##################################
process_this_frame = True
print("\n\n\t\tWelcome to Deep-Learning Face Recognition Model")
#identify the images whos value end with .jpg only
image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('images/'))
image_filenames = sorted(image_filenames)

#images path
paths_to_images = ['images/' + x for x in image_filenames]
face_encodings = []

#Encode the images located in the folder to thier respective numpy arrays
for path_to_image in paths_to_images:
	face_encodings_in_image = get_face_encodings(path_to_image)

	face_encodings.append(face_encodings_in_image[0])
print("\n\n\t\tTrained on the Images available in the folder")

#get their names
names = [x[:-4] for x in image_filenames]

#start recognizing for individual frames
while 1:
	ret, frame = webcam.read()
	small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

	if process_this_frame:
		detected_face = face_detector(small_frame, 1)

		frame_face_encodings = get_new_face_encodings(small_frame, detected_face)

		length = len(frame_face_encodings)
		#print(length)
		face_names = []
		if length == 0:
			continue
		elif length == 1:
			match = find_match(face_encodings, names, frame_face_encodings[0])
		else:
			x = 0
			while x < length:
				match = find_match(face_encodings, names, frame_face_encodings[x])
				#print('More images: ' + match)
				face_names.append(match)
				x = x + 1
		#face_names.append(match)
	process_this_frame = not process_this_frame

	cv2.putText(frame, 'Found {} face in the frame'.format(len(detected_face)), (50, 40), 
													font, 0.5, (255, 0, 0), 1, cv2.CV_AA)
	if length == 1:
		for face_rect in detected_face:
			top = face_rect.top()*2
			left = face_rect.left()*2
			right = face_rect.right()*2
			bottom = face_rect.bottom()*2

			draw_rectangle(top, left, right, bottom)
			cv2.putText(frame, '{}'.format(match.capitalize()), (left - 10, top - 20), font, .7, 
																		(66, 152, 243), 2, cv2.CV_AA)
	else:
		for face_rect, name in zip(detected_face, face_names):
			top = face_rect.top()*2
			left = face_rect.left()*2
			right = face_rect.right()*2
			bottom = face_rect.bottom()*2

			draw_rectangle(top, left, right, bottom)
			cv2.putText(frame, name.capitalize(), (left - 10, top - 20), font, .7, (66, 152, 243), 2, cv2.CV_AA)

	cv2.putText(frame, 'Press \'q\' to exit', (10,20), font, 0.5, (255, 0, 0), 1, cv2.CV_AA)
	cv2.imshow("Live Feed for Recognition", frame)
	#print("done")
	if cv2.waitKey(1) & 0xFF == ord('q'):
		print("\n\n\t\tThankyou!")
		print("\n\n")
		break

cv2.destroyAllWindows()
webcam.release()