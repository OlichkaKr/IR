import sys
import os
import cv2
from PIL import Image
import numpy as np
from scipy import misc

# from model import getImage, sess, mylogger
from gdrive import getImagesFromS3
from digits import recognize
#from fs_emu import getImagesFromGDrive, createImageFromGDriveObject

def extractDigitsFromImage (image):

    # init sample template
    sample_right = cv2.imread(os.path.dirname(__file__)+"/sample_right.jpg", cv2.IMREAD_GRAYSCALE)
    
    img = cv2.imread('image.bmp')         
            
    # rotate 90 
    # h, w, k = img.shape
    # if w>h:            
    #     M = cv2.getRotationMatrix2D((w/2,h/2),270,1)
    #     img = cv2.warpAffine(img,M,(w,h))
    #     # crop black sides
    #     img = img[0:h, (w-h)/2:h+(w-h)/2]
    #     h, w, k = img.shape
    
    # resize
    # img = cv2.resize(img, dsize=(480, 640), interpolation = cv2.INTER_AREA)
    # h, w, k = img.shape
    
    # sample_G4 = cv2.imread(os.path.dirname(__file__)+"/sample_G4.jpg")        
    # mask = np.zeros(img.shape[:2],np.uint8)
    # mask[0:int(h/2),0:w] = 255
    # mean = self.findFeature(sample_G4, img, mask)
    # if not mean:
    #     return False
    
    # # caclulate "center" point coordinates
    # x_center = int(mean[0])
    # y_center = int(mean[1]) + 25
    
    # img = misc.imresize(img, 0.5)
    h, w, k = img.shape
    # make grayscale image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # blur
    kernel = np.ones((3,3),np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # gray.save('text')

#         cv2.imshow('img',gray)
#         key = cv2.waitKey(0)
#         if key==1048603 or key==27:
#             sys.exit()    
    
    # search for edges using Canny
    edges = cv2.Canny(gray, 100, 200)
    test_image = Image.fromarray(edges)
    test_image.save('test.png')
    
#         cv2.imshow('img',edges)
#         key = cv2.waitKey(0)
#         if key==1048603 or key==27:
#             sys.exit()    
    
    # detect lines
    
    h, w, k = img.shape
    # if None == lines:
    #     mylogger.warn("No lines found")
    #     return False 
    
#         cv2.imshow('img',img)
#         key = cv2.waitKey(0)
#         if key==1048603:
#             sys.exit()    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 100, 10)
    print(lines)
    x_top_l = 24
    y_top_l = 44

    x_top_r = 136
    y_top_r = 46

    x_bot_r = 138
    y_bot_r = 68

    x_bot_l = 28
    y_bot_l = 65
    img = img[y_top_l:y_bot_r, x_bot_l:x_bot_r]
    cv2.imwrite('test.bmp', img)
    # detect nearest for center horisontal lines
#     rho_below = rho_above = np.sqrt(h*h+w*w)
#     line_above = None
#     line_below = None
#     for line in lines:
#         rho,theta = line[0]
#         sin = np.sin(theta)
#         cos = np.cos(theta)
        
#         # discard not horisontal
#         if (sin<0.75):
#             continue

#         # # calculate rho for line parallel to current line and passes through the "center" point             
#         # rho_center = x_center*cos + y_center*sin

#         x_top = 194
#         y_top = 304
#         rho_top = x_top*cos + y_top*sin

#         x_bot = 201
#         y_bot = 446
#         rho_bot = x_bot*cos + y_bot*sin
        
#         # compare line with nearest line above
#         if rho_top>=rho and rho_top-rho<=rho_above:
#             rho_above = rho_top-rho
#             line_above = {"rho":rho, "theta":theta, "sin":sin, "cos":cos}
            
# #                 x0 = cos*rho
# #                 y0 = sin*rho
# #                 x1 = int(x0 + 1000*(-sin))
# #                 y1 = int(y0 + 1000*(cos))
# #                 x2 = int(x0 - 1000*(-sin))
# #                 y2 = int(y0 - 1000*(cos))    
# #                 cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        
#         # compare line with nearest line below
#         if rho_bot<=rho and rho-rho_bot<=rho_below:
#             rho_below = rho-rho_bot
#             line_below = {"rho":rho, "theta":theta, "sin":sin, "cos":cos}
            
# #                 x0 = cos*rho
# #                 y0 = sin*rho
# #                 x1 = int(x0 + 1000*(-sin))
# #                 y1 = int(y0 + 1000*(cos))
# #                 x2 = int(x0 - 1000*(-sin))
# #                 y2 = int(y0 - 1000*(cos))    
# #                 cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# #         cv2.imshow('img',img)
# #         key = cv2.waitKey(0)
# #         if key==1048603 or key==27:
# #             sys.exit()    
# # 
# #         x0 = line_above["cos"]*line_above["rho"]
# #         y0 = line_above["sin"]*line_above["rho"]
# #         x1 = int(x0 + 1000*(-line_above["sin"]))
# #         y1 = int(y0 + 1000*(line_above["cos"]))
# #         x2 = int(x0 - 1000*(-line_above["sin"]))
# #         y2 = int(y0 - 1000*(line_above["cos"]))    
# #         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
# #             
# #         x0 = line_below["cos"]*line_below["rho"]
# #         y0 = line_below["sin"]*line_below["rho"]
# #         x1 = int(x0 + 1000*(-line_below["sin"]))
# #         y1 = int(y0 + 1000*(line_below["cos"]))
# #         x2 = int(x0 - 1000*(-line_below["sin"]))
# #         y2 = int(y0 - 1000*(line_below["cos"]))    
# #         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# #                     
# #         cv2.imshow('img',img)
# #         key = cv2.waitKey(0)
# #         if key==1048603:
# #             sys.exit()     
     
#     # check result 
# #     if line_below==None or line_above==None:
# #         mylogger.warn("No lines found")
# # #             cv2.imshow('img',img)
# # #             key = cv2.waitKey(0)
# # #             if key==1048603:
# # #                 sys.exit()            
# #         return False 
     
#     # make lines horizontal
#     M = cv2.getRotationMatrix2D((w/2,(line_below["rho"]-line_above["rho"])/2+line_above["rho"]),
#                                 (line_above["theta"]+line_below["theta"])/2/np.pi*180-90,1)
#     img = cv2.warpAffine(img,M,(w,h))

#     # crop image
#     img = img[line_above["rho"]-int(w/2*line_above['cos']):line_below["rho"]-int(w/2*line_below['cos']), 0:w]
#     h, w, k = img.shape
    
# #         sample_star = cv2.imread(os.path.dirname(__file__)+"/sample_star.jpg")        
# #         mask = np.zeros(img.shape[:2],np.uint8)
# #         mask[0:h,0:int(w/4)] = 255
# #         mean = self.findFeature(sample_star, img, mask, silent=True)
    
# #         # crop image
# #         if mean:
# #             img = img[0:h, mean[0]:w]
# #             h, w, k = img.shape
    
# # #         cv2.imshow('img',img)
# # #         key = cv2.waitKey(0)
# # #         if key==1048603:
# # #             sys.exit()    

# #         # binarize using adaptive threshold
# #         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
# #         thres = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2) 

# #         # try find arrow using findFeature
# #         sample_arr = cv2.imread(os.path.dirname(__file__)+"/sample_arr.jpg")        
# #         mask = np.zeros(img.shape[:2],np.uint8)
# #         mask[0:h,int(w/2):w] = 255
# #         mean = self.findFeature(sample_arr, img, mask, silent=True)

# #         if mean:
# #             x_right = mean[0]-12
# #         else:
# #             # match sample_right template
# #             res = cv2.matchTemplate(thres,sample_right,cv2.TM_CCORR_NORMED)
# #             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# #             x_right = max_loc[0]-4

# # #         cv2.circle(thres, (x_right, 5), 5, (0,0,255))
# # #         cv2.imshow('img',thres)
# # #         key = cv2.waitKey(0)
# # #         if key==1048603:
# # #             sys.exit()        
    
# #         kernel = np.ones((5,5),np.uint8)
# #         thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    
# #         # search for left edge position
# #         x_left=0
# #         while x_left<w :
# #             if thres[h/2,x_left]==0:
# #                 break
# #             x_left+=1
#     x_left = 194
#     x_right = 851

#     # crop image
#     img = img[:, x_left:x_right]
#     h, w, k = img.shape
    
# #         cv2.imshow('img',img)
# #         key = cv2.waitKey(0)
# #         if key==1048603:
# #             sys.exit()        
    
# #         # check ratio
# #         if float(w)/float(h)<5.5 or float(w)/float(h)>10:
# #             mylogger.warn("Image has bad ratio: %f" % (float(w)/float(h)))
# # #             cv2.imshow('img',img)
# # #             key = cv2.waitKey(0)
# # #             if key==1048603:
# # #                 sys.exit()
# #             return False    
    
# #         cv2.imshow('img',img)
# #         key = cv2.waitKey(0)
# #         if key==1048603:
# #             sys.exit()     
    
#     # self.digits_img = img
    return True

def splitDigits (image):

    # if None == self.digits_img:
    #     if not self.extractDigitsFromImage():
    #         return False

    img = cv2.imread('test.bmp') 
    img = misc.imresize(img, 1.3)
    h, w, k = img.shape
    print(h, w, k)
    
    # split to digits
    for i in range(1,8):
        digit = img[0:h, (i-1)*w/7:i*w/7]
        dh, dw, dk = digit.shape
        # binarize each digit
        digit_gray = cv2.cvtColor(digit,cv2.COLOR_BGR2GRAY)
        
        digit_bin = cv2.adaptiveThreshold(digit_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,-5)

        # remove some noise
        # kernel = np.ones((2,2),np.uint8)
        # digit_bin = cv2.morphologyEx(digit_bin, cv2.MORPH_OPEN, kernel)
        
#             cv2.imshow("digit",digit_bin)
#             k = cv2.waitKey(0)
#             if k==1048603:
#                 sys.exit()            
        test_image = Image.fromarray(digit_bin)
        test_image.save(str(i) + '.png')
        # find contours
        contours, hierarhy = cv2.findContours(digit_bin.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        #mylogger.debug("Crop digit %d" % i)
        
        # analyze contours
        biggest_contour = None
        biggest_contour_area = 0
        for cnt in contours:
            M = cv2.moments(cnt)

            # centroid
            cx = M['m10']/M['m00'] if M['m00'] else 0
            cy = M['m01']/M['m00'] if M['m00'] else 0
            
            # mylogger.debug("Center: %f,%f" % (cx/dw,cy/dh))
            
            # digit must be in the center
            if cx/dw<0.15 or cx/dw>0.85 or cy/dh<0.25 or cy/dh>0.75:
                continue
            
            # mylogger.debug("Area: %d, perimeter: %d" % (cv2.contourArea(cnt),cv2.arcLength(cnt,True)))

            # skip very small area contours
            if cv2.contourArea(cnt)<17:
                continue
            # skip very small perimeter contours
            if cv2.arcLength(cnt,True)<30:
                continue

            # identify biggest contour
            if cv2.contourArea(cnt)>biggest_contour_area:
                biggest_contour = cnt
                biggest_contour_area = cv2.contourArea(cnt)
                biggest_contour_cx = cx
                biggest_contour_cy = cy
        
        # if biggest contour not found, mark digit unknown
        if biggest_contour is None:
            print("Digit %d: no biggest contour found" % i)
            # digit.save(str(i))
            # digit = self.dbDigit(i, digit_bin)
            # digit.markDigitForManualRecognize (use_for_training=False)
            # mylogger.warn("Digit %d: no biggest contour found" % i)
            continue    
        
        # filter digit image using biggest contour    
        mask = np.zeros(digit_bin.shape,np.uint8)
        cv2.drawContours(mask,[biggest_contour],0,255,-1)
        digit_bin = cv2.bitwise_and(digit_bin,digit_bin,mask = mask)
        
#             cv2.imshow("digit",digit_bin)
#             k = cv2.waitKey(0)
#             if k==1048603:
#                 sys.exit()        
        
        # caclulate bounding rectangle
        rw = dw/2.0
        rh = dh/1.4
        if biggest_contour_cy-rh/2 < 0:
            biggest_contour_cy = rh/2
        if biggest_contour_cx-rw/2 < 0:
            biggest_contour_cx = rw/2
        
        # crop digit
        digit_bin = digit_bin[int(biggest_contour_cy-rh/2):int(biggest_contour_cy+rh/2), int(biggest_contour_cx-rw/2):int(biggest_contour_cx+rw/2)]
        
        digit_base_h = 24
        digit_base_w = 16

        # adjust size to standart
        digit_bin = cv2.resize(digit_bin,(digit_base_w, digit_base_h))
        digit_bin = cv2.threshold(digit_bin, 128, 255, cv2.THRESH_BINARY)[1]
        
        # # save to base
        # digit = self.dbDigit(i, digit_bin)
        test_image = Image.fromarray(digit_bin)
        test_image.save('c_' + str(i) + '.png')
    return True

if __name__ == '__main__':

    image = getImagesFromS3()
    # image = Image.open('image.bmp')     
    extractDigitsFromImage(image)
    splitDigits(image)
    recognize()
    # try to recognize image
    # if image.identifyDigits():
    #     mylogger.info("Result is %s" % 200)
        
