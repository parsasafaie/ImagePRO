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
    linecolor_bgr : tuple
        Color in BGR format (for OpenCV functions) of bounding boxes. Default is (0, 0, 255) which represents red in BGR.
    linewidth : int
        Defines the thickness of the lines for bounding boxes around detected faces. Default is 2 pixels.
    textcolor : str
        Specifies the color of the text displaying the confidence score. Default color is "blue".
    textcolor_bgr : tuple
        Text color in BGR format, used in OpenCV video feeds. Default is (255, 0, 0), which represents blue in BGR.
    textsize : int
        Determines the font size of the text displaying confidence scores. Default size is 10 points.
    textshape : str
        Specifies the font style for confidence text, with possible values of "normal" or "bold". Default is "normal".
    textboxcolor : str
        Color of the background box behind the confidence score text, enhancing text readability. Default is "blue".
    textboxcontrast : float
        Transparency level of the text box background, ranging from 0 (transparent) to 1 (opaque). Default is 0.3.
    
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

    Example Usage
    -------------
    ```python
    # Instantiate the ImagePRO class
    image_pro = ImagePRO()

    # Display faces with bounding boxes and confidence in an image
    image_pro.display_faces('sample.jpg', want_confidence=True)

    # Save detected face images to a folder, with optional confidence scores
    image_pro.save_faces_images('sample.jpg', folder_name='detected_faces', want_confidence=True)

    # Save coordinates of detected faces to a text file
    image_pro.save_faces_coordinates('sample.jpg', 'face_coordinates.txt', want_confidence=True)

    # Process and display video feed with face detection and confidence scores
    image_pro.process_video_feed('sample_video.mp4', want_confidence=True)
    ```
    """

    def __init__(self) -> None:
        """
        Initializes the ImagePRO instance with customizable default attributes for visual styling.

        The `__init__` method configures visual attributes such as bounding box line color, width, text color, text box 
        color, and transparency for displaying face detection results. These settings affect the appearance of bounding 
        boxes and confidence score text in all methods that display or save face data.

        Notes
        -----
        Attributes can be adjusted after instantiation to customize visual styling across methods like `display_faces`
        and `process_video_feed`.
        """
        self.linecolor = 'red'
        self.linecolor_bgr = (0, 0, 255)
        self.linewidth = 2
        self.textcolor = 'blue'
        self.textcolor_bgr = (255, 0, 0)
        self.textsize = 10
        self.textshape = 'normal'
        self.textboxcolor = 'blue'
        self.textboxcontrast = 0.3
        self.mtcnn = MTCNN(keep_all=True)

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

        Example
        -------
        ```python
        image_pro.display_faces("example.jpg", want_confidence=True)
        ```

        Notes
        -----
        - This method uses MTCNN to detect faces and provides an option to display confidence scores.
        - Bounding boxes are drawn around each detected face with customizable color and thickness, 
          based on `linecolor` and `linewidth` attributes.
        - When `want_confidence` is True, the confidence score text appears above each bounding box, 
          with font size and color set by `textsize` and `textcolor`.
        """
        image = Image.open(img_path).convert("RGB")
        boxes, probs = self.mtcnn.detect(image, landmarks=False)

        if boxes is not None:
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("arial", self.textsize) if self.textshape == "normal" else ImageFont.load_default()

            for i, box in enumerate(boxes):
                draw.rectangle(box.tolist(), outline=self.linecolor, width=self.linewidth)
                if want_confidence and probs is not None:
                    text = f"{probs[i]:.2f}"
                    text_size = draw.textsize(text, font=font)
                    text_position = (box[0], box[1] - text_size[1])
                    draw.rectangle([text_position, (text_position[0] + text_size[0], text_position[1] + text_size[1])], fill=self.textboxcolor)
                    draw.text(text_position, text, fill=self.textcolor, font=font)

        image.show()

    def save_faces_images(self, image_path: str, folder_name: str = "faces", want_confidence: bool = False) -> None:
        """
        Detects faces in an image, saving each face as a separate file in a specified folder.

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

        Example
        -------
        ```python
        image_pro.save_faces_images("group_photo.jpg", folder_name="output_faces", want_confidence=True)
        ```

        Notes
        -----
        - Each detected face is saved as a separate file in the specified folder. Face images are named `face_n.jpg`,
          where `n` is the index of the face in the image.
        - If `want_confidence` is True, a `.txt` file accompanies each face image, containing the confidence score.
        - This method is useful for applications requiring cropped face images, such as training datasets.
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        image = Image.open(image_path).convert("RGB")
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
        Detects faces in an image and saves the coordinates of each bounding box to a text file.

        Parameters
        ----------
        img_path : str
            Path to the image file for face detection.
        destination_path : str
            Path to the output text file where coordinates will be saved.
        want_confidence : bool, optional
            If True, includes the confidence score for each detected face in the text file. Default is False.

        Returns
        -------
        None

        Example
        -------
        ```python
        image_pro.save_faces_coordinates("image.jpg", "coordinates.txt", want_confidence=True)
        ```

        Notes
        -----
        - This method is useful for logging or analyzing face positions within an image.
        - Bounding box coordinates are saved in (x1, y1, x2, y2) format, representing the top-left and bottom-right
          corners of each bounding box.
        - If `want_confidence` is enabled, each bounding box entry includes the confidence score for that detection.
        """
        image = Image.open(img_path).convert("RGB")
        boxes, probs = self.mtcnn.detect(image)

        with open(destination_path, "w") as f:
            if boxes is not None:
                for i, box in enumerate(boxes):
                    box_coords = f"Face {i + 1}: {box.tolist()}"
                    if want_confidence and probs is not None:
                        box_coords += f", Confidence: {probs[i]:.2f}"
                    f.write(box_coords + "\n")

    def process_video_feed(self, video_path: str = 0, want_confidence: bool = False) -> None:
        """
        Processes a video feed or file for real-time face detection, displaying bounding boxes and confidence scores.

        Parameters
        ----------
        video_path : str, optional
            Path to the video file for processing. If set to 0, the method will open the default webcam. Default is 0.
        want_confidence : bool, optional
            If True, displays confidence scores for each detected face in the video stream. Default is False.

        Returns
        -------
        None

        Notes
        -----
        - Press 'q' to quit the video stream.
        - Bounding boxes and confidence scores are displayed in real-time using OpenCV. Bounding box colors, thickness, 
          and text styles are customizable via instance attributes.
        """
        video = cv2.VideoCapture(video_path)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            boxes, probs = self.mtcnn.detect(pil_frame)

            if boxes is not None:
                for i, box in enumerate(boxes):
                    box = [int(coord) for coord in box]
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), self.linecolor_bgr, self.linewidth)
                    if want_confidence and probs is not None:
                        confidence_text = f"{probs[i]:.2f}"
                        cv2.putText(frame, confidence_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.textcolor_bgr, 1)

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()
