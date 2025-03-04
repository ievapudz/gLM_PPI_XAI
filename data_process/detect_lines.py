import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from optparse import OptionParser
import cv2

MATRIX_PATH = "./outputs/entropy_matrices/"
MATRIX_SUFFIX = "fastEntropy.npy"
PNG_PATH = "./outputs/visualisations/"
PNG_SUFFIX = "fastEntropy.png"

def get_protein_length(length_list, complex_id):
    # Read the length list from the CSV file
    lengths = {}
    with open(length_list, 'r') as file:
        for line in file:
            identifier, length = line.strip().split(',')
            lengths[identifier] = int(length)
    
    # Get the first protein id from the concatenated complex id
    first_protein_id = complex_id.split('_')[0]

    # Return the length of the first protein
    return lengths[first_protein_id]

def outlier_count(upper_right_quadrant, mode="P95", n=3):
    if mode == "IQR":
        Q1 = np.percentile(upper_right_quadrant, 25)
        Q3 = np.percentile(upper_right_quadrant, 75)
        IQR = Q3-Q1
        threshold = Q3+1.5*IQR

    elif mode == "mean_stddev":
        m = np.mean(upper_right_quadrant)
        s = np.std(upper_right_quadrant)
        threshold = m+n*s

    elif mode == "P95":
        threshold = max(np.percentile(upper_right_quadrant, 95), 0.2)
        print(threshold)

    count_above_threshold = np.sum(upper_right_quadrant > threshold)

    return count_above_threshold

def detect_Hough(edges, output_stripe, output_diag):
    lines = cv2.HoughLines(edges, 1, np.pi/180, int(len(orig_matrix)*0.1))

    if lines is not None:
        for rho, theta in lines[:, 0]:

            angle = np.degrees(theta)

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            if(angle == 0 or angle == 90):
                cv2.line(output_stripe, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if(angle == 135 or angle == 45):
                cv2.line(output_diag, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if lines is not None and len(lines) > 0:
        print(f"Detected {len(lines)} lines")
        line_detected = True

    else:
        print("No lines detected")
        line_detected = False

def detect_HoughP(edges, output_stripe, output_diag):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(len(orig_matrix)*0.1), minLineLength=int(len(orig_matrix)*0.1), maxLineGap=5)

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0:  # Strictly vertical line
                vertical = True
                horizontal = False
                diagonal = False
            elif dy == 0:  # Strictly horizontal line
                vertical = False
                horizontal = True
                diagonal = False
            else:
                slope = dy / dx
                vertical = abs(slope) > 10  # High slope = vertical
                horizontal = abs(slope) < 0.1  # Low slope = horizontal
                diagonal = 0.5 <= abs(slope) <= 2
            
            if vertical or horizontal:
                cv2.line(output_stripe, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if diagonal:
                cv2.line(output_diag, (x1, y1), (x2, y2), (255, 0, 0), 2)

parser = OptionParser()

parser.add_option("--input", "-i", dest="input",
    help="complex ID, whose entropy matrix should be plotted.")

parser.add_option("--length-list", "-l", dest="length_list",
    help="list with lengths of the proteins.")

(options, args) = parser.parse_args()

matrix = np.load(f"{MATRIX_PATH}/{options.input}_{MATRIX_SUFFIX}")
len1 = get_protein_length(options.length_list, options.input)

matrix = np.clip(matrix, 0, 1)

ignore_len1 = int(len1*0.1)
ignore_len2 = int((matrix.shape[0]-len1)*0.1)

# Detecting the PPI signal in upper right quadrant of matrix
upper_right_quadrant = matrix[ignore_len1:len1-ignore_len1, len1+ignore_len2:-ignore_len2]

print("Number of outliers: ", outlier_count(upper_right_quadrant, mode="P95"))

matrix = (matrix*255).astype(np.uint8)
matrix = 255 - matrix
orig_matrix = matrix
matrix = matrix[:len1,len1:]

# Apply edge detection (Canny or Prewitt)
edges = cv2.Canny(matrix, 50, 150)  # Auto-detect edges

# Draw detected lines on the original image
output_stripe = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)  
output_diag = cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)  

detect_Hough(edges, output_stripe, output_diag)

# Display the original matrix and the edge-detected matrices
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

# Original matrix
ax1.imshow(orig_matrix, cmap="Reds", vmin=0, vmax=255)
ax1.axvline(x=len1-0.5, color='black', linestyle='--', linewidth=0.5)
ax1.axhline(y=len1-0.5, color='black', linestyle='--', linewidth=0.5)
ax1.set_title("Original Matrix")

ax2.imshow(output_stripe, cmap="grey", vmin=0, vmax=255)
#ax2.axvline(x=len1-0.5, color='red', linestyle='--', linewidth=0.5)
#ax2.axhline(y=len1-0.5, color='red', linestyle='--', linewidth=0.5)
ax2.set_title("Output")

ax3.imshow(output_diag, cmap="grey", vmin=0, vmax=255)
#ax3.axvline(x=len1-0.5, color='red', linestyle='--', linewidth=0.5)
#ax3.axhline(y=len1-0.5, color='red', linestyle='--', linewidth=0.5)
ax3.set_title("Output diag.")
    

plt.tight_layout()
plt.show()