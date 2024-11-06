import cv2
import numpy as np
import tensorflow as tf
from Sudokuhelper import *
import streamlit as st
import os
# image_path='1.jpg'
# height_image=450
# width_image=450
# Test Score = 0.03514572232961655
# Test Accuracy = 0.9882772564888
model=initializemodel('model_trained_new3.h5')
# Preparing the image
# img=cv2.imread(image_path)
def SudokuSolver(image_path,output_image_path):
    # image_path='1.jpg'
    height_image=450
    width_image=450
    model=initializemodel('model_trained_new3.h5')
    # Preparing the image
    img=cv2.imread(image_path)
    img=cv2.resize(img,(width_image,height_image))
    # Creating a blank image
    imgBlank=np.zeros((height_image,width_image,3),np.uint8) 
    imgThreshold=preprocess(img)

    # Finding the Contours
    imageCountours=img.copy()
    imageBigContour=img.copy()
    contours,heirarchy=cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imageCountours,contours,-1,(0,255,0),3)

    # Finding the biggest contour
    biggest, maxArea = biggest_Contour(contours) # FIND THE BIGGEST CONTOUR
    # print(biggest)
    if biggest.size != 0:
        biggest = reorder(biggest)
        # print(biggest)
        cv2.drawContours(imageBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[width_image, 0], [0, height_image],[width_image, height_image]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
        imgWarpColored = cv2.warpPerspective(img, matrix, (width_image, height_image))
        imgDetectedDigits = imgBlank.copy()
        imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgWarpColored_new = imgWarpColored.copy()

    # Finding each box
        imageSolvedDigits=imgBlank.copy()
        boxes=split_boxes(imgWarpColored)
        dimensions=boxes[0].shape #Get the dimensions of the boxes
        # print(dimensions)
        # print(imgWarpColored.shape)
        
        
        predictions=np.zeros((81,2),dtype=object)
        for index,box in enumerate(boxes):
            if index<81:
                predicted_class,prediction_probabilities=predict_image(box,model)
                predictions[index,0]=predicted_class
                predictions[index,1]=prediction_probabilities
        # print(predictions)
        
        for i in range(predictions.shape[0]):
            if predictions[i, 1] < 0.5:
                predictions[i, 0] = 0
        #Since in predicted_class all wrong predictions of 0's have probability of less than 0.5
        # print("Predictions:")
        # print(predictions) # an numpy array of shape (81,2)
        
        board = []
        board_new=[]
        for i in range(predictions.shape[0]):
            board.append(predictions[i, 0])
            board_new.append(predictions[i,0])
        print("Board:")
        print(board) #array
        board = np.array(board).reshape(9, 9) # Reshape to 9x9
        board_new=np.array(board_new,dtype=int).reshape(9,9)
        board_solution=solve(board_new)
        # print("Solution:")
        # print(board_solution) # an numpy array of shape (9,9)
        board_solution=np.array(board_solution,dtype=int).reshape(81,1)  # Reshape to 81x1
        board=np.array(board,dtype=int)
        # # board_solution=board_solution.reshape(81,1)
        board=board.reshape(81,1)
        display_board = np.zeros((81,), dtype=int)  # You can change the dtype if needed
        # print("Initial Board:\n", board)
        board_solution=board_solution.reshape((9,9))
        print("Board Solution:\n", board_solution)

        
        for i in range(len(board.flatten())):  
            if board.flatten()[i] != 0:  
                display_board[i] = 0  
            else:
                display_board[i] = board_solution.flatten()[i]  
        display_board = display_board.reshape(9, 9)

        # print("Display Board:\n", display_board)
        Display_Sudoku(imgWarpColored,display_board)
        # cv2.imshow("Sudoku",imgWarpColored)
        # cv2.waitKey(0)
        # return imgWarpColored
        cv2.imwrite(output_image_path, imgWarpColored)
    return output_image_path


# Streamlit app
st.set_page_config(page_title="Sudoku Solver", page_icon="🧩", layout="wide")
st.title("Sudoku Solver")
st.sidebar.title("Upload Sudoku Image")
st.sidebar.markdown("Upload a Sudoku puzzle image and get the solved Sudoku!")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    input_image_path = "temp_sudoku.jpg"
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Path for output image
    output_image_path = "solved_sudoku.jpg"

    # Solve the Sudoku
    solved_image_path = SudokuSolver(input_image_path, output_image_path)
    
    # Display the solved Sudoku image
    st.image(solved_image_path, caption="Solved Sudoku", use_column_width=True)
else:
    st.info("Please upload an image of a Sudoku puzzle.")