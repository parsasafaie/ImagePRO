import torch
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt

class ImagePRO:
    """
    A class for detecting and displaying faces in images using the MTCNN model from the facenet_pytorch library.

    Attributes:
    ----------
    linecolor : str
        The color of the box drawn around detected faces (default is "red").
    linewidth : int
        The width of the lines used to draw the face boxes (default is 2).
    """

    def __init__(self) -> None:
        """
        Initializes the ImagePRO object by setting default values for linecolor and linewidth.

        Attributes initialized:
        - linecolor: str : The color of the face bounding boxes (default: "red").
        - linewidth: int : The width of the bounding box lines (default: 2).
        
        Returns:
        --------
        None
        """
        self.linecolor = "red"
        self.linewidth = 2

    def show_faces(self, img_path: str) -> None:
        """
        Detects faces in an image and displays the image with rectangles drawn around detected faces.

        Parameters:
        ----------
        img_path : str
            The file path of the image to be processed.

        Functionality:
        --------------
        - Loads the image.
        - Detects faces using the MTCNN model.
        - Draws bounding boxes around detected faces.
        - Displays the image with the bounding boxes using matplotlib.

        Returns:
        --------
        None
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=True, device=device)

        img = Image.open(img_path)
        boxes, probs = mtcnn.detect(img)

        plt.imshow(img)
        ax = plt.gca()

        if boxes is not None:
            for box in boxes:
                try:
                    rect = plt.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        fill=False, color=self.linecolor, linewidth=self.linewidth)
                    ax.add_patch(rect)
                except:
                    print("Some variables are undefined. Please initialize the object using objname.__init__() or set them manually.")
                    break

            plt.show()

    def write_faces(self, img_path: str, destination_path: str) -> None:
        """
        Detects faces in an image and writes their bounding box coordinates to a text file.

        Parameters:
        ----------
        img_path : str
            The file path of the image to be processed.
        destination_path : str
            The file path where the face coordinates will be saved.

        Functionality:
        --------------
        - Loads the image.
        - Detects faces using the MTCNN model.
        - Writes the coordinates of the bounding boxes to the specified file.
        - If no faces are detected, writes a message indicating no faces were found.

        Returns:
        --------
        None
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=True, device=device)

        img = Image.open(img_path)
        boxes, _ = mtcnn.detect(img)

        with open(destination_path, 'w') as f:
            f.write("Face coordinates detected by ImagePRO:\n")
            if boxes is not None:
                for box in boxes:
                    box_str = ','.join([str(coord) for coord in box])
                    f.write(f"{box_str}\n")
            else:
                f.write("No faces detected.")