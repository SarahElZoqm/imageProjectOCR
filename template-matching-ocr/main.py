# python main.py --image images/credit_card_01.png

import cv2
from utils import *

digits = preprocess_template()
Contours,inputImage = GettingContour()
locations = get_groups_locations(Contours)
output = extract_groups(locations,digits,inputImage)

#Show image after detection
CardType = {"4": "Visa","5": "MasterCard"}
#print output to screen
print("Credit Card Type: {}".format(CardType[output[0]]))
print("Credit Card #: {}".format("".join(output)))
#Save output to database file

with open("database.txt", "a") as myfile:
   for num in output:
   	myfile.write(num)
   myfile.write("\tType: "+CardType[output[0]]+"\n")

myfile.close()

cv2.imshow("Image", inputImage)
cv2.waitKey(0)
