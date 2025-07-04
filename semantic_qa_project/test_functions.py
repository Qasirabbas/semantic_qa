import easyocr
import cv2
import matplotlib.pyplot as plt

# Load the image using OpenCV (optional visualization)
image_path = "2521749700673_.pic_hd.jpg"
image = cv2.imread(image_path)

# Check if image loaded
if image is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

# Display the image (optional)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Input Image")
plt.axis("off")
plt.show()

# Run EasyOCR
reader = easyocr.Reader(['en', 'ch_sim'])
results = reader.readtext(image_path)

# Print detected texts
for bbox, text, conf in results:
    print(f"Detected text: {text} (Confidence: {conf:.2f})")
