import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinterdnd2 import TkinterDnD, DND_FILES

root = TkinterDnD.Tk()
root.title("Handwritten Number Recognition System")
root.geometry("1000x700")

drop_box = tk.Frame(root, width=500, height=300, relief="solid", borderwidth=2, background="gainsboro")

drop_box.pack(padx=20, pady=20)
drop_box.pack_propagate(False)

image_label = tk.Label(drop_box, text="Insert or drop file here", font=("Arial", 14), background="gainsboro")
image_label.pack(expand=True, fill="both")

root.mainloop()