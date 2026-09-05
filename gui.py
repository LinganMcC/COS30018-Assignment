import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinterdnd2 import TkinterDnD, DND_FILES

def display_image(file_path):
    try:
        image = Image.open(file_path)

        # Resize while keeping aspect ratio
        image.thumbnail((400, 300))

        photo = ImageTk.PhotoImage(image)

        image_label.config(image=photo, text="")
        image_label.image = photo  # Keep reference so image isn't garbage collected

    except Exception as e:
        image_label.config(
            image="",
            text=f"Could not load image\n{e}"
        )

def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg"),
        ]
    )

    if file_path:
        display_image(file_path)

def drop_image(event):
    # Drag and drop path can sometimes be wrapped in {}
    file_path = event.data.strip("{}")

    # Get file extension
    extension = file_path.lower().split(".")[-1]

    if extension in ("png", "jpg", "jpeg"):
        display_image(file_path)

# Initialize window
root = TkinterDnD.Tk()
root.title("Handwritten Number Recognition System")
root.geometry("1000x700")

# Drop box
drop_box = tk.Frame(root, width=500, height=300, relief="solid", borderwidth=2, background="gainsboro")
drop_box.pack(padx=20, pady=20)
drop_box.pack_propagate(False)

#Drop box label/Image display
image_label = tk.Label(drop_box, text="Insert or drop file here", font=("Arial", 14), background="gainsboro")
image_label.pack(expand=True, fill="both")

# Make label and drop box clickable
image_label.bind("<Button-1>", lambda event: select_image())
drop_box.bind("<Button-1>", lambda event: select_image())

# Drag and drop support
drop_box.drop_target_register(DND_FILES)
drop_box.dnd_bind("<<Drop>>", drop_image)
image_label.drop_target_register(DND_FILES)
image_label.dnd_bind("<<Drop>>", drop_image)

root.mainloop()