from insightface.app import FaceAnalysis
import cv2
import numpy as np

from ImagePRO.utils.image import Image
from ImagePRO.utils.result import Result

# Constants
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_PROVIDER = "CPUExecutionProvider"


def compare_faces(*, 
    image_1: Image | None = None, 
    image_2: Image | None = None, 
    app=None
    ) -> Result:
    """
    Compare two face images using FaceAnalysis embeddings.

    This function calculates the cosine similarity between the embeddings 
    of two detected faces to check if they belong to the same person.

    Parameters
    ----------
    image_1 : Image
        Image instance (BGR data expected) to process.
    image_2 : Image
        Image instance (BGR data expected) to process.
    app (FaceAnalysis, optional): Pre-loaded FaceAnalysis model instance. 
        If not provided, it will be initialized inside the function.

    Returns
    -------
    Result
        Result where `image` is None and `data` is the result(T/F). except if no face is detected in either image, then `data` is None and meta has error info.

    Raises
    ------
    ValueError
        If images are not instances of Image.
    FileNotFoundError
        If image paths are invalid or images cannot be loaded.
    """
    if not isinstance(image_1, Image) or not isinstance(image_2, Image):
        raise ValueError("'image_1' and 'image_2' must be an instance of Image.")
    
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
    
    if image_1.source_type == 'path':
        path_1 = image_1.path
    else:
        path_1 = 'tmp1.jpg'
        cv2.imwrite(path_1, image_1._data)

    if image_2.source_type == 'path':
        path_2 = image_2.path
    else:
        path_2 = 'tmp2.jpg'
        cv2.imwrite(path_2, image_2._data)

    # Load and preprocess both images
    img1 = load_rgb(path_1)
    img2 = load_rgb(path_2)

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are invalid or the images could not be loaded.")

    # Detect faces in both images
    faces1 = app.get(img1)
    faces2 = app.get(img2)

    # Ensure a face was detected in both images
    if not faces1 or not faces2:
        return Result(image=None, data=None, meta={"source": (image_1, image_2), "operation": "compare_faces", "error": "No face detected in one or both images"})

    # Get the embeddings of the first detected face in each image
    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    # Compute cosine similarity between the embeddings
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # Return match result based on threshold
    return Result(image=None, data=True if sim > DEFAULT_SIMILARITY_THRESHOLD else False, meta={"source": (image_1, image_2), "operation": "compare_faces", "similarity": sim, "threshold": DEFAULT_SIMILARITY_THRESHOLD})
