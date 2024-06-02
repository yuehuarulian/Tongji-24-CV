import cv2
import numpy as np

# 读取黑白图像
image = cv2.imread('3.png', cv2.IMREAD_GRAYSCALE)

# 对图像进行阈值化，分割黑色和白色区域
_, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 对白色区域进行腐蚀操作，去除白色噪点
kernel = np.ones((5,5), np.uint8)
eroded = cv2.erode(thresholded, kernel, iterations=1)

# 对黑色区域进行膨胀操作，填充黑色噪点
dilated = cv2.dilate(eroded, kernel, iterations=1)

# 将腐蚀和膨胀后的图像进行合并
result = cv2.bitwise_and(thresholded, dilated)

# 显示结果图像
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
