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
    textcolor : str
        The color of the text used for displaying confidence scores (default is "red").
    textsize : int
        The font size of the confidence score text (default is 10).
    textshape : str
        The font weight/style of the confidence score text (default is "normal").
    textboxcolor : str
        The background color of the text box for confidence scores (default is "red").
    textboxcontrast : float
        The transparency level (alpha) of the text box background (default is 0.5).
    """

    def __init__(self) -> None:
        """
        Initializes the ImagePRO object by setting default values for linecolor, linewidth, and text-related attributes.

        Attributes initialized:
        - linecolor: str : The color of the face bounding boxes (default: "red").
        - linewidth: int : The width of the bounding box lines (default: 2).
        - textcolor: str : The color of the confidence score text (default: "red").
        - textsize: int : The font size of the confidence score text (default: 10).
        - textshape: str : The font weight/style of the confidence score text (default: "normal").
        - textboxcolor: str : The background color of the text box (default: "red").
        - textboxcontrast: float : The transparency level (alpha) of the text box background (default: 0.5).

        Returns:
        --------
        None
        """
        self.linecolor = 'red'
        self.linewidth = 2
        self.textcolor = 'red'
        self.textsize = 10
        self.textshape = 'normal'
        self.textboxcolor = 'red'
        self.textboxcontrast = 0.5

    def show_faces(self, img_path: str, want_confidence: bool = False) -> None:
        """
        Detects faces in an image and displays the image with rectangles drawn around detected faces.

        Parameters:
        ----------
        img_path : str
            The file path of the image to be processed.
        want_confidence : bool, optional
            If True, the confidence scores of the detected faces will be displayed (default is False).

        Functionality:
        --------------
        - Loads the image from the specified `img_path`.
        - Detects faces using the MTCNN model.
        - Draws bounding boxes around detected faces.
        - Optionally displays confidence scores for each detected face if `want_confidence` is True.
        - Displays the image with the bounding boxes and confidence scores using matplotlib.

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
            for i, box in enumerate(boxes):
                try:
                    rect = plt.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        fill=False, color=self.linecolor, linewidth=self.linewidth)
                    ax.add_patch(rect)

                    if want_confidence and probs is not None:
                        confidence = probs[i]
                        ax.text(box[0], box[1] - 10, f'{confidence:.2f}',
                                color=self.textcolor, fontsize=self.textsize, weight=self.textshape,
                                bbox=dict(facecolor=self.textboxcolor, alpha=self.textboxcontrast))
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("Some variables are undefined. Please initialize the object using objname.__init__() or set them manually.")
                    break

            plt.axis('off')  # Optional: Hide axes for better visualization
            plt.show()

    def write_faces(self, img_path: str, destination_path: str, want_confidence: bool = False) -> None:
        """
        Detects faces in an image and writes their bounding box coordinates to a text file.

        Parameters:
        ----------
        img_path : str
            The file path of the image to be processed.
        destination_path : str
            The file path where the face coordinates will be saved.
        want_confidence : bool, optional
            If True, the confidence scores of the detected faces will also be saved (default is False).

        Functionality:
        --------------
        - Loads the image from the specified `img_path`.
        - Detects faces using the MTCNN model.
        - Writes the coordinates of the bounding boxes for each detected face to the specified `destination_path`.
        - If `want_confidence` is True, writes the confidence scores of the detected faces along with the coordinates.
        - Coordinates are written as comma-separated values in the format: "x1,y1,x2,y2,confidence" when `want_confidence` is True.
        - If no faces are detected, writes a message indicating "No faces detected" in the file.

        Returns:
        --------
        None
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=True, device=device)

        img = Image.open(img_path)
        boxes, probs = mtcnn.detect(img)

        with open(destination_path, 'w') as f:
            f.write("Face coordinates detected by ImagePRO:\n")
            if boxes is not None:
                for i, box in enumerate(boxes):
                    box_str = ','.join([str(coord) for coord in box])
                    
                    if want_confidence and probs is not None:
                        confidence = probs[i]
                        f.write(f"{box_str},{confidence}\n")
                    else:
                        f.write(f"Coordinates: {box_str}\n")
            else:
                f.write("No faces detected.")
