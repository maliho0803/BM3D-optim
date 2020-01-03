import cv2
import math
import numpy
import glob


def PSNR(img1, img2):
    D = numpy.array(img1 - img2, dtype=numpy.int64)
    D[:, :] = D[:, :]**2
    RMSE = D.sum()/img1.size
    psnr = 10*math.log10(float(255.**2)/RMSE)
    return psnr

if __name__ == "__main__":

    img_paths = glob.glob("/Users/zhoumi/Downloads/copression/*.jpg")

    indexs = [3,4,5]

    for img_path in img_paths:
        print(img_path)
        for index in indexs:
            dst_path = img_path[:-4] + '_' + str(index) + '.png'
            img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(dst_path, cv2.IMREAD_GRAYSCALE)
            psnr = PSNR(img2, img1)
            print ("The PSNR between the two img of the two is %f" % psnr)

        # img1 = cv2.imread("original 2D4F.bmp", cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread("Final2.jpg", cv2.IMREAD_GRAYSCALE)
        # psnr = PSNR(img1, img2)
        # print ("The PSNR between the two img of the two is %f" % psnr)

