import cv2 as cv
import numpy as np 

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (0, 0, 0)

blank = np.zeros((500, 500, 3), dtype = 'uint8')

# cv.imshow("blank", blank)
# blank[50:100, 50:100] = 0, 255, 0

cv.rectangle(blank, (150, 150), (250, 250), GREEN, 2)

cv.circle(blank, (200, 98), 50, RED, 2)

cv.line(blank, (175, 250), (125, 350), RED, 2)

cv.line(blank, (225, 250), (300, 350), RED, 2)

cv.putText(blank, "Funny Man", (250, 400), cv.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)
cv.imshow("Final", blank)



cv.waitKey(0)