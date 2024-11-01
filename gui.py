from tkinter import Tk, filedialog, Label, Button
from edge_detection1 import EdgeDetection
import matplotlib.pyplot as plt

class ImageApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Dynamic Programming Edge Detection")
        
        self.label = Label(master, text="Chọn ảnh để bắt đầu:")
        self.label.pack(pady=10)
        
        self.btn_open = Button(master, text="Chọn Ảnh", command=self.open_image)
        self.btn_open.pack(pady=5)

        self.figure, self.ax = plt.subplots()
        self.image_label = None
        self.edge_detection = EdgeDetection()

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            self.process_image(file_path)
    
    def process_image(self, image_path):
        self.edge_detection.process_image(path=image_path, ax=self.ax)

root = Tk()
app = ImageApp(root)
root.mainloop()
