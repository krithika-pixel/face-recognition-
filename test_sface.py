import cv2

from sface_Engine import SFaceEngine


# -----------------------------
# 1. Load SFace model
# -----------------------------

model_path = "models/face_recognition_sface.onnx"

engine = SFaceEngine(model_path)

print("SFace model loaded successfully!")


# -----------------------------
# 2. Load face image
# -----------------------------

image = cv2.imread("test_face.jpg")

if image is None:
    print("Could not load image!")
    exit()

print("Image loaded successfully!")


# -----------------------------
# 3. Load face detector
# -----------------------------

detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet.onnx",
    "",
    (320, 320)
)


# Set image size
height, width = image.shape[:2]

detector.setInputSize((width, height))


# -----------------------------
# 4. Detect face
# -----------------------------

_, faces = detector.detect(image)


if faces is None:
    print("No face detected!")
    exit()


print("Face detected!")


# -----------------------------
# 5. Take first detected face
# -----------------------------

face = faces[0]


# -----------------------------
# 6. Generate SFace embedding
# -----------------------------

embedding = engine.get_embedding(
    image,
    face
)



print("Embedding generated successfully!")

print("Embedding shape:", embedding.shape)

print("First 10 values:")
print(embedding[:10])