import face_recognition
import cv2
from skimage import io

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = io.imread("images/Chandrakanth.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
chad_image = io.imread("test/chad.jpg")
chad_image_enco = face_recognition.face_encodings(chad_image)[0]

cv2.namedWindow("Live Feed for Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Feed for Recognition", 800, 400)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
	ret, frame = video_capture.read()

	# Resize frame of video to 1/2 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

	# Only process every other frame of video to save time
	if process_this_frame:
		# Find all the faces and face encodings in the current frame of video
		face_locations = face_recognition.face_locations(small_frame)
		face_encodings = face_recognition.face_encodings(small_frame, face_locations)

		face_names = []
		for face_encoding in face_encodings:
			# See if the face is a match for the known face(s)
			match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
			match2 = face_recognition.compare_faces([chad_image_enco], face_encoding)
			name = "Unknown"

			if match[0]:
				name = "ChandraKanth"
			if match2[0]:
				name = "Chad Smith"

			face_names.append(name)

	process_this_frame = not process_this_frame


	# Display the results
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		top *= 2
		right *= 2
		bottom *= 2
		left *= 2

		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Draw a label with a name below the face
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	# Display the resulting image
	cv2.imshow("Live Feed for Recognition", frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()