import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model

def preprocess(image):
    image_Gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_Blur=cv2.GaussianBlur(image_Gray,(5,5),1)
    imgThreshold=cv2.adaptiveThreshold(image_Blur,255,1,1,11,2)
    return imgThreshold

def biggest_Contour(contours):
    biggest=np.array([])
    max_area=0
    for i in contours:
        area=cv2.contourArea(i)
        if area>50:
            perimeter=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*perimeter,True)
            if area>max_area and len(approx)==4:
                biggest=approx
                max_area=area
    return biggest,max_area

def reorder(biggest):
    MyPoints=biggest.reshape((4,2))
    MyPointsNew=np.zeros((4,1,2),np.int32)
    add=MyPoints.sum(1)
    MyPointsNew[0]=MyPoints[np.argmin(add)]
    MyPointsNew[3]=MyPoints[np.argmax(add)]
    diff=np.diff(MyPoints,axis=1)
    MyPointsNew[1]=MyPoints[np.argmin(diff)]
    MyPointsNew[2]=MyPoints[np.argmax(diff)]
    return MyPointsNew



def split_boxes(image):
    rows=np.vsplit(image,9)
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def initializemodel(path):
    model=tf.keras.models.load_model(path)
    return model

def preProcessing(image):
    image=cv2.equalizeHist(image)
    image=image/255.0
    return image


def predict_image(img, model):
    # Load and preprocess the image
    # img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))  # Resize image to 32x32 pixels
    img = preProcessing(img)  # Apply the preprocessing
    img = img.reshape(1, 32, 32, 1)  # Reshape to fit model input shape

    # Predict the class
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Get class with highest probability
    predicted_probabilities = np.amax(prediction)
    return predicted_class,predicted_probabilities

def solve(board):
    for i in range(0,board.shape[0]):
        for j in range(0,board.shape[1]):
            if(board[i][j]==0):
                for num in range(1,10):
                    if(isValid(board,i,j,num)):
                        board[i][j]=num
                        result=solve(board)
                        if result is not None:
                            return result
                        board[i][j]=0
                return None
    return board

def isValid(board,row,col,num):
    if num in board[row,:]:
        return False
    if num in board[:,col]:
        return False
    row_start=(row//3)*3
    col_start=(col//3)*3
    for i in range(3):
        for j in range(3):
            if board[row_start + i, col_start + j] == num:
                return False
    return True

def displaynumbers(image,numbers,color=(0,255,0)):
    
    secH=int(image.shape[0]//9)
    secW=int(image.shape[1]//9)
    for x in range(0,9):
        for y in range(0,9):
            if numbers[(y*9)+x]!=0:
                cv2.putText(image,str(numbers[(y*9)+x]),(x*secW+int(secW/2)-10,int(y+0.8)*secH),cv2.FONT_HERSHEY_COMPLEX,2,color,2,cv2.LINE_AA)
    return image

def Display_Sudoku(image,board_answers):
    font=cv2.FONT_HERSHEY_SIMPLEX
    thickness=2
    font_scale=1
    if image.shape[0]<450 or image.shape[1]<450:
        raise ValueError("Check if image is minimum 450*450 pixels")
    for i in range(0,450,50):
        cv2.line(image,(i,0),(i,450),(0,0,0),2)
        cv2.line(image,(0,i),(450,i),(0,0,0),2)
    for row in range(0,9):
        for col in range(0,9):
            num=board_answers[row][col]
            if num!=0:
                text=str(num)
                text_size=cv2.getTextSize(text,font,font_scale,thickness)[0]
                text_x=col*50 + (50-text_size[0])//2
                text_y=row*50 + (50+text_size[1])//2
                cv2.putText(image,text,(text_x,text_y),font,font_scale,(0,0,0),thickness)
    return image