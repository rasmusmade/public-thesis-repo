import cv2
import numpy as np

img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.putText(img, "Hello from WSLg!", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("WSLg Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
