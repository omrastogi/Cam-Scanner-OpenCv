import cv2
import numpy as np


def align(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    return hnew


def check(h):
	print (h)
	for i in range(len(h)):
		for j in range(len(h)):
			if h[i][0][0]==h[j][0][0] and  h[i][0][1]==h[j][0][1]:
				return 0
	return 1


def view(tag,img):
	cv2.imshow(tag,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



image=cv2.imread("t3.jpg")   #read in the image
image=cv2.resize(image,(1300,800)) #resizing because opencv does not work well with bigger images
orig=image.copy()

view("img",image)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #RGB To Gray Scale
# view("Title",gray)

# view(gray)

blurred=cv2.GaussianBlur(gray,(3,3),3)  #(5,5) is the kernel size and 0 is sigma that determines the amount of blur
# view("Blur",blurred)


edged=cv2.Canny(blurred,0,255)  #30 MinThreshold and 50 is the MaxThreshold
# view("Canny",edged)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

opening = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

# view("open",opening)



contours,hierarchy=cv2.findContours(opening,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model

print(len(contours))

contours=sorted(contours,key=cv2.contourArea,reverse=True)

if len(contours)>600:
	blurred=cv2.GaussianBlur(gray,(3,3),2)
	view("blur2",blurred)
	edged=cv2.Canny(blurred,0,255)  
	dil = cv2.dilate(edged, kernel, iterations = 10)
	view("Canny1",dil)
	contours,hierarchy=cv2.findContours(dil,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
contours=sorted(contours,key=cv2.contourArea,reverse=True)




#the loop extracts the boundary contours of the page
for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.01*p,True)
    print (cv2.contourArea(c))


    if len(approx)==4 and cv2.contourArea(c)> 400000:
        print ("yo")
        target=approx
        break

approx= align(target) #find endpoints of the sheet
# print ("area",cv2.contourArea(c))

x =400

pts=np.float32([[0,0],[x,0],[x,x],[0,x]])  #map to 800*800 target window
# print (approx)
for points in list(approx):
	orig = cv2.circle(orig, tuple(points), 3, (0,255,0), 2)
# view("dot",orig) 
op=cv2.getPerspectiveTransform(approx,pts)  #get the top or bird eye view effect
dst=cv2.warpPerspective(orig,op,(x,x))

view("Scanned",dst)