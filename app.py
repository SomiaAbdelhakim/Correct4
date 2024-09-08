import cv2
from random import randrange
import config
import numpy as np
from lib.argparser import args
from json import dumps

# read image
#img = cv2.imread('mimc.jpg')
original_img_path = args.input
img = cv2.imread(original_img_path)
h, w = img.shape[:2]

# trim 15 from bottom and 5 from right to remove partial answer and extraneous red
img = img[0:h-15, 0:w-5]

# threshold on white color
lower=(225,225,225)
upper=(255,255,255)
thresh = cv2.inRange(img, lower, upper)
thresh = 255 - thresh

# apply morphology close
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

# get contours
result = img.copy() 
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = contours[0] if len(contours) == 2 else contours[1]
# print("count:", len(contours))
# print('')
i = 1
for cntr in contours:
    M = cv2.moments(cntr)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv2.circle(result, (cx, cy), 18, (0, 255, 0), -1)
    # pt = (cx,cy)
    # print("circle #:",i, "center:",pt)
    # i = i + 1

# save results
cv2.imwrite('omr_sheet_result2.png',result)



original_img_path = 'omr_sheet_result2.png'
output_path = './output/'
choices_no_per_question = 5
questions_per_row = 4
questions_direction = 'column' # row | column


def get_correct_answers():
  answers = {1: 2, 2: 1, 3: 5, 4: 1, 5: 4, 6: 3, 7: 1, 8: 2, 9: 4, 10: 5, 11: 3, 12: 2, 13: 4, 14: 1, 15: 3, 16: 1, 17: 4, 18: 5, 19: 3, 20: 2, 21: 3, 22: 3, 23: 5, 24: 1, 25: 4, 26: 5, 27: 3, 28: 5, 29: 4, 30: 3, 31: 2, 32: 1, 33: 3, 34: 2, 35: 3, 36: 5, 37: 1, 38: 3, 39: 2, 40: 3, 41: 4, 42: 3, 43: 4, 44: 1, 45: 3, 46: 2, 47: 5, 48: 5, 49: 1, 50: 1, 51: 2, 52: 3, 53: 4, 54: 5, 55: 2, 56: 2, 57: 4, 58: 1, 59: 5, 60: 5, 61: 3, 62: 2, 63: 4, 64: 2, 65: 4, 66: 3, 67: 5, 68: 4, 69: 1, 70: 1, 71: 2, 72: 4, 73: 3, 74: 4, 75: 1, 76: 5, 77: 2, 78: 4, 79: 1, 80: 3, 81: 5, 82: 2, 83: 3, 84: 1, 85: 4, 86: 5, 87: 3, 88: 1, 89: 5, 90: 1, 91: 4, 92: 2}

  return answers

def read_image(path):
  image = cv2.imread(path)

  if image is None:
    raise Exception('Image not found')

  return image 

def clone_original_image():
  return read_image(original_img_path)

def save_image(image, name):
  cv2.imwrite(output_path + name, image)

def detect_document(image):
  return image

def preprocess_document(image):
  clone = image.copy()
  # Gray scale image
  clone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

  # Blur image to remove noise
  clone = cv2.GaussianBlur(clone, (5, 5), 0)

  # Binarize image
  clone = cv2.threshold(clone, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  save_image(clone, 'preprocessed.jpg')

  return clone 


def optimize_image_rotation(image):
  return image

def detect_all_bubbles(image):
  clone = image.copy()

  bubbles_contours = []

  # Find contours
  cnts = cv2.findContours(clone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Draw contours
  with_contours = cv2.drawContours(clone_original_image(), cnts[0], -1, (0, 0, 255), 3)
  save_image(with_contours, 'all_contours_marked.jpg')


  #Find bubbles
  for c in cnts[0]:
    (x, y, w, h) = cv2.boundingRect(c)

    # Check if the contour is a bubble
    # By checking if it's width and height are sufficient (Big enough to not select random dots)
    # And by checking if it's ratio is close to one (Circle)
    isBubble = (w >= 25 and w <= 100) and (h >= 25 and h <= 100) and (w / h >= 0.88 and w / h <= 1.12 ) 
    if isBubble:
      bubbles_contours.append(c)
  bubbles_only = cv2.drawContours(clone_original_image(), bubbles_contours, -1, (0, 0, 255), 3)
  save_image(bubbles_only, 'bubbles_contours_only.jpg')

  return bubbles_contours


# Group bubbles that belongs to the same question
def detect_bubbles_groups(all_bubbles_contours, choices_no_per_question, questions_per_row):
  # Grouped bubbles by question
  bubbles_groups = []

  # Sort bubbles by y position
  sorted_contours = sorted(all_bubbles_contours, key=lambda x: cv2.boundingRect(x)[1])

  bubbles_no_per_row = choices_no_per_question * questions_per_row

  for i in range(0, len(sorted_contours), bubbles_no_per_row):
    row_of_bubbles = sorted_contours[i:i + bubbles_no_per_row]

    # Sort bubbles by x position
    sorted_row_of_bubbles = sorted(row_of_bubbles, key=lambda x: cv2.boundingRect(x)[0])

    for i in range(0, len(sorted_row_of_bubbles), choices_no_per_question):
      question_bubbles = sorted_row_of_bubbles[i:i + choices_no_per_question]
      bubbles_groups.append(question_bubbles)

  # Sort bubbles groups by question direction
  if questions_direction == 'row':
    pass
  elif questions_direction == 'column':
    sorted_questions_groups = []
    
    for i in range(0, questions_per_row):
      for j in range(i, len(bubbles_groups), questions_per_row):
        sorted_questions_groups.append(bubbles_groups[j])

    bubbles_groups = sorted_questions_groups


  # Draw grouped bubbles
  original_document = clone_original_image()

  for g in bubbles_groups:
    random_color = (randrange(0, 255), randrange(0, 255), randrange(0, 255))
    org = cv2.drawContours(original_document, g, -1, random_color, 3)
  
  save_image(org, 'all_grouped_bubbles.jpg')

  # Examples for the first and second questions only grouped
  g1 = cv2.drawContours(clone_original_image(), bubbles_groups[0], -1, (0, 0, 255), 3)
  g2 = cv2.drawContours(clone_original_image(), bubbles_groups[1], -1, (255, 0, 0), 3)

  save_image(g1, 'grouped_bubbles_example_1.jpg')
  save_image(g2, 'grouped_bubbles_example_2.jpg')

  return bubbles_groups

def detect_marked_bubble_per_group(image, grouped_bubbles, min_pixels_to_be_marked=900):
  answers = {}

  original_document = clone_original_image()

  least_total = 1000000000000000000000

  for i, g in enumerate(grouped_bubbles):
    anwser_number = None
    max_marked = None
    max_marked_contour = None

    for a_number, cnt in  enumerate(g):
      mask = np.zeros(image.shape, dtype="uint8")
      cv2.drawContours(mask, [cnt], -1, 255, -1)
      mask = cv2.bitwise_and(image, image, mask=mask)
      total_marked_pixels = cv2.countNonZero(mask)

      if total_marked_pixels < least_total:
        least_total = total_marked_pixels

      if max_marked_contour is None or total_marked_pixels >= max_marked:
        if total_marked_pixels < min_pixels_to_be_marked:
          continue

        max_marked = total_marked_pixels
        max_marked_contour = cnt
        anwser_number = a_number + 1

    answers[i + 1] = anwser_number

    if max_marked_contour is not None:
      cv2.drawContours(original_document, [max_marked_contour], -1, (0, 255, 0), 3)

  save_image(original_document, 'marked_bubbles.jpg')

  return answers


def calculate_grade(answers, correct_answers):
  grade = 0

  for key in correct_answers:
    if answers[key] == correct_answers[key]:
      grade += 1

  return grade

def main():
  try:
    image = clone_original_image()
    document = detect_document(image)
    optimized_document = optimize_image_rotation(document)
    preprocessed_document = preprocess_document(optimized_document)
    all_bubbles = detect_all_bubbles(preprocessed_document)
    grouped_bubbles = detect_bubbles_groups(all_bubbles, choices_no_per_question, questions_per_row)

    answers = detect_marked_bubble_per_group(preprocessed_document, grouped_bubbles)

    # Print answers as JSON
    print(dumps(answers))

  except Exception as e:
    print(dumps({'error': str(e)}))


if __name__ == '__main__':
  main() 