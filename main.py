import matplotlib.pyplot as plt

from face_recognition_matrix import face_recognition_matrix

# Specify the directory path
directory = 'img2'

frm = face_recognition_matrix(directory)
frm.load_images()
frm.make_matrix()

# Show the final image
plt.imshow(frm.canvas)
plt.axis('off')
plt.show()

frm.close()
