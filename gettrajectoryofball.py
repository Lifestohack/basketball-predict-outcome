from sklearn.preprocessing import MinMaxScaler
import os
import cv2
import numpy as np
path_to_read = "/run/media/luma/Blade/Users/lumi/basketball/dataset/matlab/training/hit/0/view2/"
images_path = os.listdir(path_to_read)
images = []
for image in images_path:
    image = os.path.join(path_to_read, image)
    img = cv2.imread(image)
    images.append(img)

images = np.stack(images)
single_img = images.sum(axis=0) / 700
cv2.imshow("cropped", single_img)
cv2.waitKey(0)
#cv2.imwrite("view1.png", single_img)
pass