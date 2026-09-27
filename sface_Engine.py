import cv2


class SFaceEngine:

    def __init__(self, model_path):
        self.recognizer = cv2.FaceRecognizerSF.create(
            model_path,
            ""
        )

    def get_embedding(self, image, face):
        """
        image = complete image
        face = detected face information
        """

        aligned_face = self.recognizer.alignCrop(
            image,
            face
        )

        embedding = self.recognizer.feature(
            aligned_face
        )

        return embedding.flatten()