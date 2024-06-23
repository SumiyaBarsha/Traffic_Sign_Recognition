import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, Label
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.feature import hog
from skimage import exposure
from main import *
from tkinter import *
from PIL import ImageTk, Image
import os
import re

#dictionary to label all traffic signs class.
classes = { 0:'Speed limit (15km/h)',
            1:'No Stopping',      
            2:'Go straight or right',       
            3:'No U turn',      
            4:'U turn',    
            5:'Keep Right',      
            6:'Dangerous Curve to the left',     
            7:'Go Left',    
            8:'Go Straight',     
            9:'No Horn' }

                 
def load_image():
    global img_path, img, img_display
    img_path = filedialog.askopenfilename()  
    if img_path:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_image(img_display, ax1, "Original Image")
        
        show_block_normalized_HOG_Descriptor(img)
        compute_gradients_and_show(img)
        
        # Load and preprocess test image using HOG
        global test_hog
        test_hog = preprocess_image_with_hog(img_path)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        show_hog_features(img_path, axs, 'Test Image')
        plt.show()
        
 

def classify_image():
    global templates_hog
    if 'test_hog' in globals():
        
        # Match HOG descriptors with the predefined templates
        matches = match_hog_descriptors(test_hog, templates_hog)
        best_match_index = np.argmin(matches)
        best_template_path = template_paths[best_match_index]
        best_template = cv2.imread(best_template_path, cv2.IMREAD_GRAYSCALE)
        
        print("All template paths:", template_paths)
        print("All matches:", matches)
        print("Best match index:", best_match_index)
        print("Best template path:", best_template_path)

        display_image(best_template, ax2, f'Best HOG match is template {best_match_index} with a match score of {matches[best_match_index]}')
       
        class_label = classes[best_match_index]
        label.config(text=f"Class: {class_label}")
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        show_hog_features(img_path, axs, 'Best Matched Template')
        plt.show()

def display_image(img, ax, title):
    ax.clear()
    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    ax.set_title(title)
    ax.axis('off')
    canvas.draw()



def compute_gradients_and_show(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
         image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    magnitude, orientation, grad_x, grad_y = compute_gradients(image)
    
    # Visualization
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(grad_x, cmap='gray')
    axs[1].set_title('Gradient X')
    axs[2].imshow(grad_y, cmap='gray')
    axs[2].set_title('Gradient Y')
    axs[3].imshow(magnitude, cmap='gray')
    axs[3].set_title('Magnitude')
    axs[4].imshow(orientation, cmap='gray')
    axs[4].set_title('Orientation')
    plt.show()


def show_block_normalized_HOG_Descriptor(image):
    hog_descriptor=compute_hog_Descriptor(image)
    plt.figure(figsize=(10, 4))
    plt.title('Block Normalized HOG Descriptor')
    plt.plot(hog_descriptor)
    plt.show()
    
#############################################################################
global template_paths,templates_hog
# Function to extract the numerical part from the filename
def extract_number(filename):
    parts = re.findall(r'\d+', filename)
    return int(parts[0]) if parts else float('inf')  # Return a large number if no number is found

template_directory = "G:/Image_project/data/template/"
# List files, sort them using the key, and create full paths
template_paths = [os.path.join(template_directory, f) 
                  for f in sorted(os.listdir(template_directory), key=extract_number) 
                  if f.endswith(('.png', '.jpg', '.jpeg'))]

templates_hog = [preprocess_image_with_hog(p) for p in template_paths]




root = tk.Tk()

root.title("Traffic Sign Classification")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, columnspan=4)

btn_load = tk.Button(root, text="Load Image", command=load_image,width=20, height=1, bg='blue' , fg= 'white')
btn_load.grid(row=1, column=0)

btn_classify = tk.Button(root, text="Classify", command=classify_image,width=20, height=1, bg= 'green', fg = 'white')
btn_classify.grid(row=1, column=1)


btn_exit = tk.Button(root, text="Exit", command=root.destroy,width=20, height=1, bg='red', fg='white')
btn_exit.grid(row=1, column=3)

label = Label(root, font=('arial', 15, 'bold'), height=2, bg='white', fg='black')
label.grid(row=2, column=0,columnspan=4, sticky='ew')

root.mainloop()



cv2.waitKey(0)
cv2.destroyAllWindows()
