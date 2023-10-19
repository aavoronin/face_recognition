import os

import face_recognition
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from main import directory


class face_recognition_matrix:
    def __init__(self, folder):
        """folder containing group of images"""
        self.imgs = []
        self.folder = folder

    def load_images(self):
        # Collect all files in the directory
        files = os.listdir(self.folder)

        self.imgs = []
        # Print the list of files
        for file in files:
            self.imgs.append(os.path.join(self.folder, file))

        self.imgs.sort()

    def compare_faces(self, image1, image2):
        # Find the face encodings in the images
        encoding1 = face_recognition.face_encodings(image1)[0]
        encoding2 = face_recognition.face_encodings(image2)[0]

        return self.face_distance(self, encoding1, encoding2)

    def face_distance(self, encoding1, encoding2):
        # Compare the face encodings
        face_distance = face_recognition.face_distance([encoding1], encoding2)
        # Calculate the face similarity percentage
        similarity_percentage = (1 - face_distance[0]) * 100
        return round(similarity_percentage, 1)

    def make_matrix(self):
        n = len(self.imgs)
        self.m = np.zeros((n,n),dtype='float')

        self.run_recognition(n)
        print(self.m)
        self.draw_marix()
        self.canvas.save(os.path.join("results", directory + '.png'))

    def draw_marix(self):
        # Create dataframe
        df = pd.DataFrame(self.m, index=self.imgs, columns=self.imgs)
        # Define image size
        cell_size = 64
        # Create a blank canvas for the final image
        canvas_width = cell_size * (df.shape[1] + 1)
        canvas_height = cell_size * (df.shape[0] + 1)
        self.canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
        # Resize and paste row index images on the left
        for i, row_img in enumerate(df.index):
            img = Image.open(row_img).resize((cell_size, cell_size))
            self.canvas.paste(img, (0, (i + 1) * cell_size))
        # Resize and paste column index images on the top
        for i, col_img in enumerate(df.columns):
            img = Image.open(col_img).resize((cell_size, cell_size))
            self.canvas.paste(img, ((i + 1) * cell_size, 0))
        # Draw dataframe values as text onto the image
        # from PIL import Image, ImageDraw, ImageFont
        draw = ImageDraw.Draw(self.canvas)
        font = ImageFont.truetype("timesbd.ttf", 24)  # Replace with your desired font and size
        # text_width, text_height = draw.multiline_textsize('ABC', font=font)
        for i, row_img in enumerate(df.index):
            for j, col_img in enumerate(df.columns):
                if i == j:
                    continue
                cell_value = df.loc[row_img, col_img]
                text = f'{cell_value:.1f}'
                # text_width, text_height = draw.multiline_textsize(text, font=font)
                text_width = 50
                text_height = 24
                text_position = ((j + 1) * cell_size + cell_size // 2 - text_width // 2,
                                 (i + 1) * cell_size + cell_size // 2 - text_height // 2)
                draw.text(text_position, text, fill='black', font=font)

    def run_recognition(self, n):
        face_encodings = []
        for i in range(n):
            print(self.imgs[i])
            try:
                image = face_recognition.load_image_file(self.imgs[i])
                face_encodings.append(face_recognition.face_encodings(image)[0])
            except:
                face_encodings.append(None)
        for i in range(n):
            print(self.imgs[i])
            face_encodings1 = face_encodings[i]
            self.m[i, i] = 100.0
            for j in range(i + 1, n):
                print(i, j)
                face_encodings2 = face_encodings[j]
                try:
                    self.m[j, i] = self.m[i, j] = self.face_distance(face_encodings1, face_encodings2)
                except Exception as e:
                    pass

    def close(self):
        pass
