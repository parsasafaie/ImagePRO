from facenet_pytorch import MTCNN
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class ImagePRO:
    """
    A comprehensive utility class for face detection, display, and image/video processing using the MTCNN model.
    
    This class includes methods to detect faces within images and videos, display bounding boxes with confidence scores,
    save detected faces as individual image files, and log face coordinates to text files. The MTCNN model provides 
    accurate face detection, and the class attributes allow extensive customization of appearance for bounding boxes 
    and text. 

    Attributes
    ----------
    linecolor : str
        Specifies the color of the bounding box drawn around each detected face in the image or video. Default is "red".
    linewidth : int
        Defines the thickness of the lines for bounding boxes around detected faces. Default is 2 pixels.
    textcolor : str
        Specifies the color of the text displaying the confidence score. Default color is "white".
    textsize : int
        Determines the font size of the text displaying confidence scores. Increased for better visibility.
    textfont : str
        Specifies the font style for confidence text. If set to "default", uses the default font; otherwise, attempts 
        to load the specified font. Default is "default".
    mtcnn : MTCNN
        An instance of the MTCNN model used for face detection.
    
    Methods
    -------
    display_faces(img_path: str, want_confidence: bool = False) :
        Detects and displays faces within an image, marking each face with a bounding box and optionally adding a 
        confidence score above the face. The image is displayed in a separate viewer with the option to close it manually.
    save_faces_images(image_path: str, folder_name: str = "faces", want_confidence: bool = False) :
        Detects faces within an image and saves each detected face as an individual image in a specified folder.
        Optionally saves a text file with confidence scores alongside each face image.
    save_faces_coordinates(img_path: str, destination_path: str, want_confidence: bool = False) :
        Writes the bounding box coordinates of detected faces to a text file, with an option to include confidence scores.
        This provides a useful record for further analysis or logging face positions within images.
    process_video_feed(video_path: str = 0, want_confidence: bool = False) :
        Detects faces in a live video feed or from a video file, displaying bounding boxes and confidence scores 
        in real-time. The stream updates continuously until the user presses 'q' to exit.
    """
    
    def __init__(self) -> None:
        """
        Initializes the ImagePRO instance with customizable default attributes for visual styling.

        The `__init__` method configures visual attributes such as bounding box line color, width, text color and text box 
        color for displaying face detection results. These settings affect the appearance of bounding 
        boxes and text in all methods that display or save face data.

        Notes
        -----
        Attributes can be adjusted after instantiation to customize visual styling across methods like `display_faces`
        and `process_video_feed`.
        """
        self.linecolor = 'red'
        self.linewidth = 2
        self.textcolor = 'white'
        self.textsize = 160
        self.textfont = 'default'
        self.mtcnn = MTCNN(keep_all=True)

    def _get_font(self) -> ImageFont:
        """
        Helper method to load the specified font. Uses the default font if `textfont` is set to "default"
        or falls back to the default font in case the specified font is unavailable.
        
        Returns
        -------
        ImageFont
            The font object to use for drawing text on the image.
        """
        if self.textfont == 'default':
            return ImageFont.load_default()
        try:
            return ImageFont.truetype(self.textfont, self.textsize)
        except IOError:
            print(f"Warning: Font '{self.textfont}' is unavailable. Using default font.")
            return ImageFont.load_default()

    def display_faces(self, img_path: str, want_confidence: bool = False) -> None:
        """
        Detects faces in an image, displaying bounding boxes and optional confidence scores.

        Parameters
        ----------
        img_path : str
            Path to the image file for face detection. Accepted formats include JPG, PNG, etc.
        want_confidence : bool, optional
            If True, displays the confidence score above each detected face, providing a visual indicator 
            of detection accuracy. Default is False.

        Returns
        -------
        None
            Opens a viewer displaying the image with bounding boxes and confidence scores, if enabled.
        """
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print("Image path not found.")
            return
        
        boxes, probs = self.mtcnn.detect(image, landmarks=False)

        if boxes is not None:
            draw = ImageDraw.Draw(image)
            font = self._get_font()

            for i, box in enumerate(boxes):
                draw.rectangle(box.tolist(), outline=self.linecolor, width=self.linewidth)
                if want_confidence and probs is not None:
                    text = f"{probs[i]:.2f}"
                    
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_x = int(box[0])
                    text_y = int(box[1]) - (text_bbox[3] - text_bbox[1]) - 20

                    if text_x < 0:
                        text_x = 0
                    if text_y < 0:
                        text_y = 0

                    text_box_x1 = text_x + (text_bbox[2] - text_bbox[0]) + 20
                    text_box_y1 = text_y + (text_bbox[3] - text_bbox[1]) + 20

                    if text_box_x1 >= text_x and text_box_y1 >= text_y:
                        draw.rectangle([text_x, text_y, text_box_x1, text_box_y1], fill=(0, 0, 0, 200))
                        draw.text((text_x + 10, text_y + 10), text, fill=self.textcolor, font=font)

        image.show()

    def save_faces_images(self, image_path: str, folder_name: str = "faces", want_confidence: bool = False) -> None:
        """
        Detects faces in an image and saves each face as a separate file in a specified folder.

        Parameters
        ----------
        image_path : str
            Path to the image file for processing.
        folder_name : str, optional
            Directory to store cropped face images. If the folder does not exist, it will be created.
            Default folder name is "faces".
        want_confidence : bool, optional
            If True, creates a text file with the confidence score for each detected face next to the face image. 
            Default is False.

        Returns
        -------
        None
        """
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print("Image path not found.")
            return
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        boxes, probs = self.mtcnn.detect(image)

        if boxes is not None:
            for i, box in enumerate(boxes):
                face = image.crop(box)
                face_path = os.path.join(folder_name, f"face_{i + 1}.jpg")
                face.save(face_path)

                if want_confidence and probs is not None:
                    confidence_path = os.path.join(folder_name, f"face_{i + 1}_confidence.txt")
                    with open(confidence_path, "w") as f:
                        f.write(f"Confidence: {probs[i]:.2f}")

    def save_faces_coordinates(self, img_path: str, destination_path: str, want_confidence: bool = False) -> None:
        """
        Writes the bounding box coordinates of detected faces to a text file.

        Parameters
        ----------
        img_path : str
            Path to the image file for face detection.
        destination_path : str
            Path to the output text file for saving coordinates.
        want_confidence : bool, optional
            If True, includes confidence scores in the output file next to each bounding box. Default is False.

        Returns
        -------
        None

        Notes
        -----
        The coordinates are saved in the format: x1, y1, x2, y2, [confidence].
        """
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print("Image path not found.")
            return
        boxes, probs = self.mtcnn.detect(image)

        with open(destination_path, "w") as f:
            if boxes is not None:
                for i, box in enumerate(boxes):
                    line = f"{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}"
                    if want_confidence and probs is not None:
                        line += f", {probs[i]:.2f}"
                    f.write(line + "\n")

    def process_video_feed(self, video_path: str = 0, want_confidence: bool = False) -> None:
        """
        Processes a video feed, detecting and displaying faces in real time.

        Parameters
        ----------
        video_path : str or int, optional
            Path to a video file or device index for a camera (default is 0 for the primary camera).
        want_confidence : bool, optional
            If True, displays confidence scores on the bounding boxes. Default is False.

        Returns
        -------
        None

        Notes
        -----
        The video feed will continue until the user presses 'q' to exit.
        """
        try:
            cap = cv2.VideoCapture(video_path)
        except:
            print("Video source not found.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, probs = self.mtcnn.detect(image)

            if boxes is not None:
                draw = ImageDraw.Draw(image)
                font = self._get_font()

                for i, box in enumerate(boxes):
                    draw.rectangle(box.tolist(), outline=self.linecolor, width=self.linewidth)
                    if want_confidence and probs is not None:
                        text = f"{probs[i]:.2f}"
                        text_bbox = draw.textbbox((0, 0), text, font=font)

                        text_x = int(box[0])
                        text_y = int(box[1]) - (text_bbox[3] - text_bbox[1]) - 20

                        if text_x < 0:
                            text_x = 0
                        if text_y < 0:
                            text_y = 0

                        text_box_x1 = text_x + (text_bbox[2] - text_bbox[0]) + 20
                        text_box_y1 = text_y + (text_bbox[3] - text_bbox[1]) + 20

                        if text_box_x1 >= text_x and text_box_y1 >= text_y:
                            draw.rectangle([text_x, text_y, text_box_x1, text_box_y1], fill=(0, 0, 0, 200))
                            draw.text((text_x + 10, text_y + 10), text, fill=self.textcolor, font=font)

            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
