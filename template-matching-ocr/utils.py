from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

#returns the templates digits dict 
def preprocess_template():
	#Read template as binary
	tem = cv2.imread("template.png")
	tem = cv2.cvtColor(tem, cv2.COLOR_BGR2GRAY)
	tem = cv2.threshold(tem, 10, 255, cv2.THRESH_BINARY_INV)[1]

	#Find contours from template image
	temCnts = cv2.findContours(tem.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	temCnts = temCnts[0] if imutils.is_cv2() else temCnts[1]
	temCnts = contours.sort_contours(temCnts, method="left-to-right")[0]
	digits_temp = {}

	#Store template for each digit in digit_temp dictthen return it
	for (i, c) in enumerate(temCnts):
		(x, y, w, h) = cv2.boundingRect(c)
		num = tem[y:y + h, x:x + w]
		num = cv2.resize(num, (57, 88))
		digits_temp[i] = num
	return digits_temp

def GettingContour():
  	arp = argparse.ArgumentParser()
	arp.add_argument("-i", "--image", required=True, help="path to input image")
	args = vars(arp.parse_args())
  	rectangularElement = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
  	squareElement= cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	inputImage = cv2.imread(args["image"])
	inputImage = imutils.resize(inputImage, width=300)
	grayScale = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
	lightRegionFinder = cv2.morphologyEx(grayScale, cv2.MORPH_TOPHAT, rectangularElement)
	gradient = cv2.Sobel(lightRegionFinder, ddepth=cv2.CV_32F, dx=1, dy=0,
	ksize=-1)
	gradient = np.absolute(gradient)
	(minimum, maximum) = (np.min(gradient), np.max(gradient))
	gradient = (255 * ((gradient - minimum) / (maximum - minimum)))
	gradient= gradient.astype("uint8")
	gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, rectangularElement)
	levelIndicator  = cv2.threshold(gradient, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	levelIndicator = cv2.morphologyEx(levelIndicator, cv2.MORPH_CLOSE, squareElement)
	Contours = cv2.findContours(levelIndicator.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	Contours = Contours[0] if imutils.is_cv2() else Contours[1]
	return Contours,inputImage

def get_groups_locations(Contours):
    locations=[]
    for (i, contour) in enumerate(Contours):

        (x, y, width, height) = cv2.boundingRect(contour)
        aspectR = width / float(height)

        if aspectR > 2.5 and aspectR < 4.0:

            if (width > 40 and width < 55) and (height > 10 and height < 20):
                locations.append((x, y, width, height))

    locations = sorted(locations, key=lambda x: x[0])
    return locations

def match_digits(group,Contours,digits):
    scores=[]
    groupOutput = []
    for c in Contours:

        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        scores = []

        for (digit, digitROI) in digits.items():

            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)


        groupOutput.append(str(np.argmax(scores)))

    return groupOutput,scores

def extract_groups(locations,digits,image):
    output=[]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (i, (g_X, g_Y, g_W, g_H)) in enumerate(locations):

        group = gray[g_Y - 5:g_Y + g_H + 5, g_X - 5:g_X + g_W + 5]
        group = cv2.threshold(group, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        digitContours = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        digitContours = digitContours[0] if imutils.is_cv2() else digitContours[1]
        digitContours = contours.sort_contours(digitContours,
                                           method="left-to-right")[0]

        groupOutput, scores=match_digits(group,digitContours,digits)

        cv2.rectangle(image, (g_X - 5, g_Y - 5),
                      (g_X + g_W + 5, g_Y + g_H + 5), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput), (g_X, g_Y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        output.extend(groupOutput)
    return output
