from insightface.app import FaceAnalysis
import cv2
import numpy as np

# Constants
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_PROVIDER = "CPUExecutionProvider"


def compare_faces(src_image_path_1, src_image_path_2, app=None):
    """
    Compare two face images using FaceAnalysis embeddings.

    This function calculates the cosine similarity between the embeddings 
    of two detected faces to check if they belong to the same person.

    Args:
        src_image_path_1 (str | Path): Path to the first image.
        src_image_path_2 (str | Path): Path to the second image.
        app (FaceAnalysis, optional): Pre-loaded FaceAnalysis model instance. 
            If not provided, it will be initialized inside the function.

    Returns:
        bool | str:
            - True  : Faces match (cosine similarity > 0.5).
            - False : Faces do not match.
            - 'No Face Detected' : If no face is found in either image.
    """
    # Prepare FaceAnalysis model if not provided
    if app is None:
        app = FaceAnalysis(
            name=DEFAULT_MODEL_NAME,
            providers=[DEFAULT_PROVIDER],
        )
        app.prepare(ctx_id=0)  # Run on CPU

    # Helper function to read an image and convert to RGB
    def load_rgb(path):
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    # Load and preprocess both images
    img1 = load_rgb(src_image_path_1)
    img2 = load_rgb(src_image_path_2)

    if img1 is None or img2 is None:
        return 'Invalid Image Path'

    # Detect faces in both images
    faces1 = app.get(img1)
    faces2 = app.get(img2)

    # Ensure a face was detected in both images
    if not faces1 or not faces2:
        return 'No Face Detected'

    # Get the embeddings of the first detected face in each image
    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    # Compute cosine similarity between the embeddings
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # Return match result based on threshold
    return True if sim > DEFAULT_SIMILARITY_THRESHOLD else False
