import torch
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class ImagePRO:
    """
    A class for detecting and displaying faces in images and videos using the MTCNN model 
    from the facenet_pytorch library.

    This class encapsulates the functionality for face detection in both static images 
    and live video streams. It provides methods to display detected faces with bounding 
    boxes and confidence scores, as well as to write the coordinates of detected faces 
    to a text file. The class supports customization of visual attributes such as box 
    colors, text styles, and transparency levels.

    Attributes
    ----------
    linecolor : str
        The color of the bounding boxes drawn around detected faces. The default value 
        is "red". This can be changed to any valid color name recognized by matplotlib.
    linecolor_bgr : tuple
        The BGR (Blue, Green, Red) representation of the bounding box color. Default is 
        (0, 0, 255), representing red in OpenCV.
    linewidth : int
        The width of the lines used to draw the face bounding boxes. The default is set 
        to 2 pixels. This value can be adjusted for thicker or thinner bounding boxes.
    textcolor : str
        The color of the text used for displaying confidence scores. The default value 
        is "blue", but can be modified to suit user preferences.
    textcolor_bgr : tuple
        The BGR representation of the text color used for confidence scores. The default 
        is (255, 0, 0), which corresponds to blue in OpenCV.
    textsize : int
        The font size for the confidence score text. The default is set to 10. Adjust 
        this value to increase or decrease the text size based on the display requirements.
    textshape : str
        The font weight/style of the confidence score text. The default is "normal", 
        but can be set to "bold" or "italic" for different text presentations.
    textboxcolor : str
        The background color of the text box that appears behind the confidence scores. 
        The default value is "blue". Users can customize this color as needed.
    textboxcontrast : float
        The transparency level (alpha) of the text box background. The default is set 
        to 0.3, allowing for a semi-transparent look. Values between 0 (fully transparent) 
        and 1 (fully opaque) can be used to adjust visibility.

    Methods
    -------
    __init__() :
        Initializes the ImagePRO object with default attribute values.
    
    display_faces(img_path: str, want_confidence: bool = False) :
        Detects faces in an image and displays it with rectangles drawn around 
        detected faces, optionally showing confidence scores.

    save_faces_coordinates(img_path: str, destination_path: str, want_confidence: bool = False) :
        Detects faces in an image and writes their bounding box coordinates to a text 
        file, optionally including confidence scores.

    process_video_feed(video_path: str = 0, want_confidence: bool = False) :
        Processes a video file or webcam feed to detect faces and display the video 
        with bounding boxes around detected faces, optionally showing confidence scores.

    Examples
    --------
    To use this class, create an instance of ImagePRO and call its methods:
    
    >>> image_pro = ImagePRO()
    >>> image_pro.display_faces('image.jpg', want_confidence=True)
    >>> image_pro.save_faces_coordinates('image.jpg', 'output.txt', want_confidence=True)
    >>> image_pro.process_video_feed('video.mp4', want_confidence=True)
    """

    def __init__(self) -> None:
        """
        Initializes an instance of the ImagePRO class with default attribute values.

        This constructor sets up various attributes that define the appearance of the 
        bounding boxes and text displayed during face detection. The attributes are 
        configurable and allow customization of the visual aspects of detected faces, 
        such as the colors, linewidth, text size, and background transparency for the 
        text box.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        The default values can be modified after instantiation to customize the appearance 
        of bounding boxes and text displayed during face detection. For example, users 
        can change colors or text size according to their specific visualization preferences.
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

    def display_faces(self, img_path: str, want_confidence: bool = False) -> None:
        """
        Detects faces in an image and displays it with rectangles drawn around 
        detected faces, optionally showing confidence scores.

        Parameters
        ----------
        img_path : str
            The file path of the image to be processed. The image should be in a format 
            supported by the PIL library (e.g., JPEG, PNG).
        want_confidence : bool, optional
            If True, the confidence scores of the detected faces will be displayed above 
            each bounding box. The default is False, meaning confidence scores will not 
            be shown.

        Returns
        --------
        None
            This method does not return any value.

        Notes
        -----
        This method uses the MTCNN model to detect faces and draws bounding boxes around 
        detected faces on the image. If `want_confidence` is set to True, it will display 
        the confidence score for each detected face, providing an indication of the 
        model's certainty in its detections. The bounding box color and text properties 
        can be customized using class attributes.

        Example
        -------
        To display an image with detected faces:
        
        >>> image_pro = ImagePRO()
        >>> image_pro.display_faces('image.jpg', want_confidence=True)
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
                    print("Some variables may be undefined. Please initialize the object using objname.__init__() or set them manually.")
                    break

            plt.axis('off')  # Optional: Hide axes for better visualization
            plt.show()

    def save_faces_coordinates(self, img_path: str, destination_path: str, want_confidence: bool = False) -> None:
        """
        Detects faces in an image and writes their bounding box coordinates to a text file, 
        optionally including confidence scores.

        Parameters
        ----------
        img_path : str
            The file path of the image to be processed. The image should be in a format 
            supported by the PIL library (e.g., JPEG, PNG).
        destination_path : str
            The file path where the coordinates of detected faces will be saved. The 
            output will be written in a text file format.
        want_confidence : bool, optional
            If True, the confidence scores of the detected faces will be included in the 
            output file. The default is False, meaning only coordinates will be written.

        Returns
        --------
        None
            This method does not return any value.

        Notes
        -----
        The coordinates are saved in the format: `x_min, y_min, x_max, y_max` for each 
        detected face. If `want_confidence` is True, each line will also include the 
        confidence score corresponding to the face detection.

        Example
        -------
        To write the coordinates of detected faces to a file:
        
        >>> image_pro = ImagePRO()
        >>> image_pro.save_faces_coordinates('image.jpg', 'output.txt', want_confidence=True)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=True, device=device)

        img = Image.open(img_path)
        boxes, probs = mtcnn.detect(img)

        with open(destination_path, 'w') as f:
            if boxes is not None:
                for i, box in enumerate(boxes):
                    if want_confidence and probs is not None:
                        confidence = probs[i]
                        f.write(f'{box[0]}, {box[1]}, {box[2]}, {box[3]}, {confidence:.2f}\n')
                    else:
                        f.write(f'{box[0]}, {box[1]}, {box[2]}, {box[3]}\n')

    def process_video_feed(self, video_path: str = 0, want_confidence: bool = False) -> None:
        """
        Processes a video file or webcam feed to detect faces and display the video 
        with bounding boxes around detected faces, optionally showing confidence scores.

        Parameters
        ----------
        video_path : str or int, optional
            The file path of the video to be processed or the index of the webcam feed. 
            The default value is 0, which typically represents the primary webcam. 
            A string can be provided for file paths.
        want_confidence : bool, optional
            If True, the confidence scores of the detected faces will be displayed above 
            each bounding box. The default is False, meaning confidence scores will not 
            be shown.

        Returns
        --------
        None
            This method does not return any value.

        Notes
        -----
        This method captures frames from the video and applies the face detection model 
        to each frame. Detected faces are highlighted in real-time, and the video stream 
        can be stopped by pressing 'q'. The bounding box color and text properties can 
        be customized using class attributes.

        Example
        -------
        To process a video file or webcam feed for face detection:
        
        >>> image_pro = ImagePRO()
        >>> image_pro.process_video_feed('video.mp4', want_confidence=True)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=True, device=device)

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = mtcnn.detect(Image.fromarray(frame_rgb))

            if boxes is not None:
                for i, box in enumerate(boxes):
                    cv2.rectangle(frame, 
                                  (int(box[0]), int(box[1])), 
                                  (int(box[2]), int(box[3])), 
                                  self.linecolor_bgr, 
                                  self.linewidth)

                    if want_confidence and probs is not None:
                        confidence = probs[i]
                        cv2.putText(frame, f'{confidence:.2f}', 
                                    (int(box[0]), int(box[1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    self.textsize / 30, 
                                    self.textcolor_bgr, 
                                    2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
