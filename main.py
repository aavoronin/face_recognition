import math
import pandas as pd
import face_recognition
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

# Specify the directory path
directory = 'img2'

# Collect all files in the directory
files = os.listdir(directory)

imgs = []
# Print the list of files
for file in files:
    imgs.append(os.path.join(directory, file))

imgs.sort()

def compare_faces(image1, image2):
    # Find the face encodings in the images
    encoding1 = face_recognition.face_encodings(image1)[0]
    encoding2 = face_recognition.face_encodings(image2)[0]

    return face_distance(encoding1, encoding2)

def face_distance(encoding1, encoding2):
    # Compare the face encodings
    face_distance = face_recognition.face_distance([encoding1], encoding2)
    # Calculate the face similarity percentage
    similarity_percentage = (1 - face_distance[0]) * 100
    return round(similarity_percentage, 1)

n = len(imgs)
m = np.zeros((n,n),dtype='float')

# Create dataframe
df = pd.DataFrame(m, index=imgs, columns=imgs)
print(df)

face_encodings = []
for i in range(n):
    print(imgs[i])
    try:
        image = face_recognition.load_image_file(imgs[i])
        face_encodings.append(face_recognition.face_encodings(image)[0])
    except:
        face_encodings.append(None)


for i in range(n):
    print(imgs[i])
    face_encodings1 = face_encodings[i]
    m[i, i] = 100.0
    for j in range(i + 1, n):
        print(i, j)
        face_encodings2 = face_encodings[j]
        try:
            m[j, i] = m[i, j] = face_distance(face_encodings1, face_encodings2)
        except Exception as e:
            pass



print(m)

# Create dataframe
df = pd.DataFrame(m, index=imgs, columns=imgs)

# Define image size
n = 64

# Create a blank canvas for the final image
canvas_width = n * (df.shape[1] + 1)
canvas_height = n * (df.shape[0] + 1)
canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')

# Resize and paste row index images on the left
for i, row_img in enumerate(df.index):
    img = Image.open(row_img).resize((n, n))
    canvas.paste(img, (0, (i + 1) * n))

# Resize and paste column index images on the top
for i, col_img in enumerate(df.columns):
    img = Image.open(col_img).resize((n, n))
    canvas.paste(img, ((i + 1) * n, 0))


# Draw dataframe values as text onto the image
#from PIL import Image, ImageDraw, ImageFont
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype("timesbd.ttf", 24)  # Replace with your desired font and size
#text_width, text_height = draw.multiline_textsize('ABC', font=font)

for i, row_img in enumerate(df.index):
    for j, col_img in enumerate(df.columns):
        if i == j:
            continue
        cell_value = df.loc[row_img, col_img]
        text = f'{cell_value:.1f}'
        #text_width, text_height = draw.multiline_textsize(text, font=font)
        text_width = 50
        text_height = 24
        text_position = ((j + 1) * n + n // 2 - text_width // 2, (i + 1) * n + n // 2 - text_height // 2)
        draw.text(text_position, text, fill='black', font=font)

# Draw dataframe values in the cross-section cells
#for i, row_img in enumerate(df.index):
#    for j, col_img in enumerate(df.columns):
#        cell_value = df.loc[row_img, col_img]
#        plt.text((j + 1) * n + n//2, (i + 1) * n + n//2, f'{cell_value:.3f}', ha='center', va='center', fontsize=8)

canvas.save(os.path.join("results", directory + '.png'))

# Show the final image
plt.imshow(canvas)
plt.axis('off')
plt.show()

# Print the dataframe
print(df)