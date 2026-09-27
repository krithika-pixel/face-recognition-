import cv2
from sface_Engine import SFaceEngine




model_path = "models/face_recognition_sface.onnx"

engine = SFaceEngine(model_path)




image = cv2.imread("test_face.jpg")

if image is None:
    print("Could not load image!")
    exit()




detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet.onnx",
    "",
    (320, 320)
)

height, width = image.shape[:2]

detector.setInputSize((width, height))




_, faces = detector.detect(image)

if faces is None:
    print("No face detected!")
    exit()


face = faces[0]




embedding = engine.get_embedding(
    image,
    face
)


print("Face recognized/tested!")

print("Embedding shape:", embedding.shape)

print("Embedding:")
print(embedding)