"""
Handwritten Number Recognition System (HNRS) - GUI
COS30018 Project Option B

Layout
------
  Left   : Settings panel (model choice, hyper-parameters, training,
           preprocessing options, segmentation options)
  Centre : Notebook with tabs
             Input        - load a file, drag & drop, or build a number
                            from a folder of single-digit images
             Preprocessed - result of the preprocessing step
             Segments     - detected digit boxes + per-digit predictions
             Training     - live loss / accuracy curves
  Bottom : Output panel (recognised number, per-digit confidence)

The section marked BACKEND PLACEHOLDERS contains simple working stand-ins
(preprocess / segment) and fake stand-ins (train / predict). Replace the
bodies of those functions with your team's real code; the GUI only depends
on their signatures.

Dependencies: pillow, numpy, matplotlib, tkinterdnd2 (optional, for drag & drop)
"""

import os
import queue
import random
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageTk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    HAS_DND = True
except ImportError:  # GUI still works, just without drag & drop
    HAS_DND = False

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
IMAGE_TYPES = ("png", "jpg", "jpeg", "bmp")
MODEL_OPTIONS = ["CNN", "MLP", "SVM", "Random Forest"]
SEGMENT_METHODS = ["Vertical projection"]  # add more as you implement them
BG = "gainsboro"
HEADER_BG = "dim gray"


# ============================================================================
# BACKEND PLACEHOLDERS  -  replace with the team's real implementations
# ============================================================================
def otsu_threshold(arr):
    """Otsu's method on a uint8 array. Returns a threshold in 0-255."""
    hist = np.bincount(arr.ravel(), minlength=256).astype(float)
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b, w_b, best, thresh = 0.0, 0.0, 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thresh = between, t
    return thresh


def preprocess(img, invert="Auto", blur=False, threshold="Otsu", thresh_value=128):
    """Return a grayscale PIL image with a WHITE digit on a BLACK background
    (the MNIST convention)."""
    gray = img.convert("L")
    if blur:
        gray = gray.filter(ImageFilter.GaussianBlur(1))
    arr = np.array(gray)

    if invert == "Yes" or (invert == "Auto" and arr.mean() > 127):
        arr = 255 - arr

    if threshold == "Otsu":
        arr = ((arr > otsu_threshold(arr)) * 255).astype(np.uint8)
    elif threshold == "Fixed":
        arr = ((arr > thresh_value) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def segment(processed, method="Vertical projection", min_width=3):
    """Split a preprocessed number image into single digits.

    Returns (digit_images, boxes): 28x28 PIL images and (x0, y0, x1, y1) boxes.
    """
    arr = np.array(processed)
    mask = arr > otsu_threshold(arr)
    cols = mask.any(axis=0)

    # find runs of "ink" columns
    runs, start = [], None
    for x, has_ink in enumerate(cols):
        if has_ink and start is None:
            start = x
        elif not has_ink and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(cols)))

    digits, boxes = [], []
    for x0, x1 in runs:
        if x1 - x0 < min_width:
            continue
        rows = np.where(mask[:, x0:x1].any(axis=1))[0]
        y0, y1 = rows[0], rows[-1] + 1
        crop = processed.crop((x0, y0, x1, y1))
        # MNIST style: fit into 20x20, centre on a 28x28 canvas
        crop.thumbnail((20, 20))
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
        digits.append(canvas)
        boxes.append((x0, y0, x1, y1))
    return digits, boxes


def train_model(model_name, params, on_epoch, stop_event):
    """PLACEHOLDER. Train `model_name` on MNIST.

    Call on_epoch(epoch, loss, accuracy) after every epoch and stop early if
    stop_event.is_set(). Return the trained model object.
    """
    loss, acc = 2.3, 0.1
    for epoch in range(1, params["epochs"] + 1):
        if stop_event.is_set():
            break
        time.sleep(0.4)  # <- real training goes here
        loss *= 0.6
        acc += (1 - acc) * 0.5
        on_epoch(epoch, loss, acc)
    return object()


def predict_digits(model_name, model, digit_images):
    """PLACEHOLDER. Return a list of (label, confidence) for each 28x28 image."""
    return [(random.randint(0, 9), random.uniform(0.5, 1.0)) for _ in digit_images]


def build_number_image(folder, number_string, spacing=6):
    """Create an image of a number by picking a random sample of each digit
    from `folder`.

    Supported folder layouts:
      folder/0/*.png ... folder/9/*.png      (one sub-folder per digit)
      folder/<digit>*.png                    (file name starts with the digit)
    """
    samples = {}
    for d in "0123456789":
        sub = os.path.join(folder, d)
        files = []
        if os.path.isdir(sub):
            files = [os.path.join(sub, f) for f in os.listdir(sub)]
        else:
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(d)]
        files = [f for f in files if f.lower().endswith(IMAGE_TYPES)]
        if files:
            samples[d] = files

    missing = sorted({c for c in number_string if c not in samples})
    if missing:
        raise ValueError(f"No images found for digit(s): {', '.join(missing)}")

    imgs = [Image.open(random.choice(samples[c])).convert("L").resize((28, 28)) for c in number_string]
    # make the background consistently dark so the result matches MNIST
    imgs = [ImageOps.invert(i) if np.array(i).mean() > 127 else i for i in imgs]
    width = sum(i.width for i in imgs) + spacing * (len(imgs) + 1)
    canvas = Image.new("L", (width, 28 + 2 * spacing), 0)
    x = spacing
    for i in imgs:
        canvas.paste(i, (x, spacing))
        x += i.width + spacing
    return canvas


# ============================================================================
# GUI
# ============================================================================
class HNRSApp:
    def __init__(self, root):
        self.root = root
        root.title("Handwritten Number Recognition System")
        root.geometry("1150x760")
        root.minsize(980, 640)

        # state
        self.original_image = None
        self.processed_image = None
        self.digit_images = []
        self.boxes = []
        self.models = {}  # model name -> trained model object
        self.history = {"loss": [], "acc": []}
        self.train_queue = queue.Queue()
        self.stop_event = threading.Event()
        self._photos = {}  # keep PhotoImage references alive

        self._init_variables()
        self._build_layout()
        self._build_settings()
        self._build_notebook()
        self._build_output()
        self._set_status("Load an image or generate one from a digit folder.")

    # ------------------------------------------------------------------ vars
    def _init_variables(self):
        self.model_var = tk.StringVar(value=MODEL_OPTIONS[0])
        self.epochs_var = tk.IntVar(value=10)
        self.batch_var = tk.IntVar(value=64)
        self.lr_var = tk.StringVar(value="0.001")
        self.invert_var = tk.StringVar(value="Auto")
        self.blur_var = tk.BooleanVar(value=False)
        self.thresh_var = tk.StringVar(value="Otsu")
        self.thresh_val_var = tk.IntVar(value=128)
        self.seg_var = tk.StringVar(value=SEGMENT_METHODS[0])
        self.minw_var = tk.IntVar(value=3)
        self.status_var = tk.StringVar()

    # ---------------------------------------------------------------- layout
    def _build_layout(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.settings = tk.Frame(self.root, width=250, relief="solid", borderwidth=2, background=BG)
        self.settings.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)
        self.settings.grid_propagate(False)

        self.center = tk.Frame(self.root)
        self.center.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        self.output = tk.Frame(self.root, height=140, relief="solid", borderwidth=2, background=BG)
        self.output.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 5))
        self.output.grid_propagate(False)

        tk.Label(self.root, textvariable=self.status_var, anchor="w").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 5)
        )

    def _section(self, parent, title):
        """Small helper: a titled sub-frame inside the settings panel."""
        box = tk.LabelFrame(parent, text=title, background=BG)
        box.pack(fill="x", padx=8, pady=4)
        return box

    def _row(self, parent, label, widget_factory):
        row = tk.Frame(parent, background=BG)
        row.pack(fill="x", padx=4, pady=2)
        tk.Label(row, text=label, width=11, anchor="w", background=BG).pack(side="left")
        w = widget_factory(row)
        w.pack(side="left", fill="x", expand=True)
        return w

    # -------------------------------------------------------------- settings
    def _build_settings(self):
        tk.Label(self.settings, text="Settings", font=("Arial", 14), anchor="w", background=HEADER_BG).pack(fill="x")

        # --- model + hyper-parameters
        box = self._section(self.settings, "Model")
        self._row(box, "Model:", lambda p: ttk.Combobox(p, textvariable=self.model_var,
                                                         values=MODEL_OPTIONS, state="readonly", width=14))
        self._row(box, "Epochs:", lambda p: ttk.Spinbox(p, from_=1, to=200, textvariable=self.epochs_var, width=8))
        self._row(box, "Batch size:", lambda p: ttk.Spinbox(p, from_=8, to=1024, increment=8,
                                                            textvariable=self.batch_var, width=8))
        self._row(box, "Learn rate:", lambda p: ttk.Entry(p, textvariable=self.lr_var, width=10))

        btns = tk.Frame(box, background=BG)
        btns.pack(fill="x", padx=4, pady=4)
        self.train_btn = ttk.Button(btns, text="Train", command=self.start_training)
        self.train_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_training, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))
        self.progress = ttk.Progressbar(box, mode="determinate")
        self.progress.pack(fill="x", padx=6, pady=(0, 6))

        # --- preprocessing
        box = self._section(self.settings, "Preprocessing")
        self._row(box, "Invert:", lambda p: ttk.Combobox(p, textvariable=self.invert_var,
                                                          values=["Auto", "Yes", "No"], state="readonly", width=10))
        self._row(box, "Threshold:", lambda p: ttk.Combobox(p, textvariable=self.thresh_var,
                                                             values=["Otsu", "Fixed", "None"], state="readonly", width=10))
        self._row(box, "Fixed value:", lambda p: ttk.Scale(p, from_=0, to=255, variable=self.thresh_val_var))
        tk.Checkbutton(box, text="Gaussian blur", variable=self.blur_var, background=BG,
                       anchor="w").pack(fill="x", padx=4)

        # --- segmentation
        box = self._section(self.settings, "Segmentation")
        self._row(box, "Method:", lambda p: ttk.Combobox(p, textvariable=self.seg_var,
                                                          values=SEGMENT_METHODS, state="readonly", width=14))
        self._row(box, "Min width:", lambda p: ttk.Spinbox(p, from_=1, to=50, textvariable=self.minw_var, width=8))

        ttk.Button(self.settings, text="Recognise Number", command=self.recognise).pack(
            fill="x", padx=12, pady=10, ipady=4
        )

    # -------------------------------------------------------------- notebook
    def _build_notebook(self):
        self.nb = ttk.Notebook(self.center)
        self.nb.pack(fill="both", expand=True)

        # ---- Input tab
        self.tab_input = tk.Frame(self.nb)
        self.nb.add(self.tab_input, text="Input")

        bar = tk.Frame(self.tab_input)
        bar.pack(fill="x", pady=5, padx=5)
        ttk.Button(bar, text="Load image...", command=self.select_image).pack(side="left", padx=2)
        ttk.Button(bar, text="Generate from digit folder...", command=self.generate_from_folder).pack(side="left", padx=2)
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="right", padx=2)

        self.drop_box = tk.Frame(self.tab_input, relief="solid", borderwidth=2, background=BG)
        self.drop_box.pack(fill="both", expand=True, padx=5, pady=5)
        self.drop_box.pack_propagate(False)
        hint = "Click here or drop an image file" if HAS_DND else "Click here to choose an image"
        self.image_label = tk.Label(self.drop_box, text=hint, font=("Arial", 14), background=BG, cursor="hand2")
        self.image_label.pack(expand=True, fill="both")

        for w in (self.drop_box, self.image_label):
            w.bind("<Button-1>", lambda e: self.select_image())
            if HAS_DND:
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self.drop_image)

        # ---- Preprocessed tab
        self.tab_pre = tk.Frame(self.nb)
        self.nb.add(self.tab_pre, text="Preprocessed")
        self.pre_label = tk.Label(self.tab_pre, text="Run recognition to see the preprocessed image.")
        self.pre_label.pack(expand=True, fill="both")

        # ---- Segments tab
        self.tab_seg = tk.Frame(self.nb)
        self.nb.add(self.tab_seg, text="Segments")
        self.seg_label = tk.Label(self.tab_seg, text="Run recognition to see the segmented digits.")
        self.seg_label.pack(fill="both", expand=True)
        self.thumb_frame = tk.Frame(self.tab_seg)
        self.thumb_frame.pack(fill="x", pady=8)

        # ---- Training tab
        self.tab_train = tk.Frame(self.nb)
        self.nb.add(self.tab_train, text="Training")
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax_loss = self.fig.add_subplot(121)
        self.ax_acc = self.fig.add_subplot(122)
        self._redraw_curves()
        self.curve_canvas = FigureCanvasTkAgg(self.fig, master=self.tab_train)
        self.curve_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------------------------------------------------------------- output
    def _build_output(self):
        tk.Label(self.output, text="Output", font=("Arial", 14), anchor="w", background=HEADER_BG).pack(fill="x")
        self.result_var = tk.StringVar(value="-")
        row = tk.Frame(self.output, background=BG)
        row.pack(fill="both", expand=True, padx=10)
        tk.Label(row, text="Recognised number:", font=("Arial", 12), background=BG).pack(side="left")
        tk.Label(row, textvariable=self.result_var, font=("Arial", 28, "bold"), background=BG).pack(side="left", padx=15)
        self.detail_var = tk.StringVar()
        tk.Label(row, textvariable=self.detail_var, justify="left", anchor="w", background=BG).pack(side="left", padx=20)

    # ----------------------------------------------------------- image input
    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", " ".join(f"*.{e}" for e in IMAGE_TYPES))],
        )
        if path:
            self.load_image(path)

    def drop_image(self, event):
        # tkinterdnd2 wraps paths containing spaces in {}
        path = event.data.strip().strip("{}")
        if path.lower().endswith(IMAGE_TYPES):
            self.load_image(path)
        else:
            self._set_status("Unsupported file type. Use: " + ", ".join(IMAGE_TYPES))

    def load_image(self, path):
        try:
            self.set_input_image(Image.open(path).copy())
            self._set_status(f"Loaded {os.path.basename(path)}")
        except Exception as e:
            self.image_label.config(image="", text=f"Could not load image\n{e}")

    def generate_from_folder(self):
        folder = filedialog.askdirectory(title="Select folder of single-digit images")
        if not folder:
            return
        number = simpledialog.askstring("Generate number", "Digits to generate (e.g. 2026):", parent=self.root)
        if not number:
            return
        if not number.isdigit():
            messagebox.showerror("Invalid input", "Please enter digits 0-9 only.")
            return
        try:
            self.set_input_image(build_number_image(folder, number))
            self._set_status(f"Generated an image of '{number}'")
        except Exception as e:
            messagebox.showerror("Could not generate image", str(e))

    def set_input_image(self, img):
        self.original_image = img
        self._show(self.image_label, "input", img, (700, 380))
        self.nb.select(self.tab_input)

    def clear(self):
        self.original_image = self.processed_image = None
        self.digit_images, self.boxes = [], []
        hint = "Click here or drop an image file" if HAS_DND else "Click here to choose an image"
        self.image_label.config(image="", text=hint)
        self.result_var.set("-")
        self.detail_var.set("")

    # ----------------------------------------------------------- recognition
    def recognise(self):
        if self.original_image is None:
            messagebox.showinfo("No image", "Load or generate an image first.")
            return
        model_name = self.model_var.get()
        if model_name not in self.models:
            messagebox.showinfo("Model not trained", f"Train the {model_name} model first.")
            return

        # Task 1 - preprocessing
        self.processed_image = preprocess(
            self.original_image, self.invert_var.get(), self.blur_var.get(),
            self.thresh_var.get(), int(self.thresh_val_var.get()),
        )
        self._show(self.pre_label, "pre", self.processed_image, (700, 380))

        # Task 2 - segmentation
        self.digit_images, self.boxes = segment(self.processed_image, self.seg_var.get(), self.minw_var.get())
        if not self.digit_images:
            self.result_var.set("?")
            self.detail_var.set("No digits found. Try different preprocessing settings.")
            return

        # Task 3 - classification
        results = predict_digits(model_name, self.models[model_name], self.digit_images)
        self._show_segments(results)

        number = "".join(str(lbl) for lbl, _ in results)
        self.result_var.set(number)
        self.detail_var.set("Confidence per digit:\n" + "  ".join(f"{l}: {c:.0%}" for l, c in results))
        self._set_status(f"Recognised {len(results)} digit(s) using {model_name}.")

    def _show_segments(self, results):
        # annotated copy of the processed image with boxes
        annotated = self.processed_image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for (x0, y0, x1, y1), (lbl, _) in zip(self.boxes, results):
            draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
        self._show(self.seg_label, "seg", annotated, (700, 300))

        # one thumbnail + prediction per digit
        for child in self.thumb_frame.winfo_children():
            child.destroy()
        for i, (digit, (lbl, conf)) in enumerate(zip(self.digit_images, results)):
            cell = tk.Frame(self.thumb_frame, relief="groove", borderwidth=1)
            cell.pack(side="left", padx=4)
            photo = ImageTk.PhotoImage(digit.resize((56, 56), Image.NEAREST))
            self._photos[f"thumb{i}"] = photo
            tk.Label(cell, image=photo).pack()
            tk.Label(cell, text=f"{lbl} ({conf:.0%})").pack()

    # -------------------------------------------------------------- training
    def start_training(self):
        try:
            params = {
                "epochs": int(self.epochs_var.get()),
                "batch_size": int(self.batch_var.get()),
                "learning_rate": float(self.lr_var.get()),
            }
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid parameters", "Check epochs, batch size and learning rate.")
            return

        model_name = self.model_var.get()
        self.history = {"loss": [], "acc": []}
        self._redraw_curves()
        self.stop_event.clear()
        self.progress.config(maximum=params["epochs"], value=0)
        self.train_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.nb.select(self.tab_train)
        self._set_status(f"Training {model_name}...")

        def worker():
            def on_epoch(epoch, loss, acc):
                self.train_queue.put(("epoch", epoch, loss, acc))

            model = train_model(model_name, params, on_epoch, self.stop_event)
            self.train_queue.put(("done", model_name, model))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll_training)

    def stop_training(self):
        self.stop_event.set()

    def _poll_training(self):
        """Tkinter is not thread-safe, so the worker only posts to a queue
        and the GUI thread drains it here."""
        finished = False
        try:
            while True:
                msg = self.train_queue.get_nowait()
                if msg[0] == "epoch":
                    _, epoch, loss, acc = msg
                    self.history["loss"].append(loss)
                    self.history["acc"].append(acc)
                    self.progress.config(value=epoch)
                    self._redraw_curves()
                    self.curve_canvas.draw_idle()
                    self._set_status(f"Epoch {epoch}: loss {loss:.4f}, accuracy {acc:.2%}")
                elif msg[0] == "done":
                    _, name, model = msg
                    self.models[name] = model
                    finished = True
        except queue.Empty:
            pass

        if finished:
            self.train_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self._set_status(f"{self.model_var.get()} training finished.")
        else:
            self.root.after(100, self._poll_training)

    def _redraw_curves(self):
        self.ax_loss.clear()
        self.ax_acc.clear()
        self.ax_loss.plot(range(1, len(self.history["loss"]) + 1), self.history["loss"])
        self.ax_acc.plot(range(1, len(self.history["acc"]) + 1), self.history["acc"])
        self.ax_loss.set_title("Loss")
        self.ax_acc.set_title("Accuracy")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_acc.set_xlabel("Epoch")
        self.fig.tight_layout()

    # --------------------------------------------------------------- helpers
    def _show(self, label, key, img, max_size):
        """Display a PIL image in a label, scaled to fit, keeping aspect ratio."""
        img = img.copy()
        img.thumbnail(max_size)
        photo = ImageTk.PhotoImage(img)
        self._photos[key] = photo  # prevent garbage collection
        label.config(image=photo, text="")

    def _set_status(self, text):
        self.status_var.set(text)


def main():
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    HNRSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()