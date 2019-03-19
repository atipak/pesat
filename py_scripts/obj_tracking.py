import cv2
import numpy as np


def nothing(x):
    pass


lowerBound0 = np.array([0, 100, 90])
upperBound0 = np.array([10, 255, 255])
lowerBound1 = np.array([160, 100, 90])
upperBound1 = np.array([179, 255, 255])
np.set_printoptions(threshold=np.nan)

# cam= cv2.VideoCapture(0)
img = cv2.imread('/home/patik/.ros/image.bmp', cv2.IMREAD_UNCHANGED)
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

# font=cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)
font = cv2.FONT_HERSHEY_SIMPLEX

# cv2.namedWindow('image')
# cv2.createTrackbar('Hmin','image',0,255,nothing)
# cv2.createTrackbar('Smin','image',0,255,nothing)
# cv2.createTrackbar('Vmin','image',0,255,nothing)
# cv2.createTrackbar('Hmax','image',0,255,nothing)
# cv2.createTrackbar('Smax','image',0,255,nothing)
# cv2.createTrackbar('Vmax','image',0,255,nothing)

while True:
    # ret, img=cam.read()
    # img=cv2.resize(img,(340,220))
    #    hmin = cv2.getTrackbarPos('Hmin','image')
    #    smin = cv2.getTrackbarPos('Smin','image')
    #    vmin = cv2.getTrackbarPos('Vmin','image')
    #    hmax = cv2.getTrackbarPos('Hmax','image')
    #    smax = cv2.getTrackbarPos('Smax','image')
    #    vmax = cv2.getTrackbarPos('Vmax','image')
    # convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # create the Mask
    #    mask=cv2.inRange(imgHSV,(hmin, smin, vmin), (hmax, smax, vmax))
    mask0 = cv2.inRange(imgHSV, lowerBound0, upperBound0)
    mask1 = cv2.inRange(imgHSV, lowerBound1, upperBound1)
    # morphology
    maskOpen = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    maskOpen1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernelOpen)
    maskClose1 = cv2.morphologyEx(maskOpen1, cv2.MORPH_CLOSE, kernelClose)
    output = img.copy()

    # detect circles in the image
    maskClose[maskClose1 == 255] = 255
    maskFinal = maskClose
    im2, conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # rgb = np.random.randint(255, size=(900,800,3),dtype=np.uint8)
    # imgCut = np.zeros(img.shape,dtype='uint8')
    # imgCut.fill(0)
    # imgCut[maskClose == 255] = 255
    #    out = np.zeros_like(img)
    #    out[maskFinal == 255] = img[maskFinal == 255]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 2,
                               param1=250, param2=50,
                               minRadius=1, maxRadius=1000)

    cv2.drawContours(img, conts, -1, (255, 0, 0), 3)
    for c in conts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) > 5:
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))
            print(cX, cY)
            print(x + w / 2, y + h / 2)
            print(w, h)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
            # cv2.putText(img, str(i+1),(x,y+h),font, 4,(255,255,255),2,cv2.LINE_AA)
    if True:
        maximal = 0
        maximal_circle = None
        print(circles)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                print(center)
                # circle center
                cv2.circle(img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                # cv2.circle(img, center, radius, (255, 0, 255), 3)
                circle_mask = np.zeros(img.shape[0:2], dtype='uint8')
                cv2.circle(circle_mask, center, radius, 255, -1)
                print(circle_mask.shape)
                print(maskFinal.shape)
                and_image = cv2.bitwise_and(maskFinal, circle_mask)
                area = np.pi * radius * radius
                nzCount = cv2.countNonZero(and_image)
                percs = nzCount / area
                if maximal < percs:
                    maximal_circle = i
        print(maximal_circle)

    # cv2.imshow("maskClose",maskClose)
    # cv2.imshow("maskOpen",maskOpen)
    # cv2.imshow("mask",mask0)
    # cv2.imshow("im2",imgCut)
    cv2.imshow("cam", img)
    # cv2.imshow("output", out)
    cv2.waitKey(0)
    break
