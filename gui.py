import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True

# GUI still works, just without drag & drop
except ImportError:  
    HAS_DND = False

# CNN backend (models/cnn/gui_backend.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cnn"))
import gui_backend as cnn_backend

# Expression parser (expression_parser.py, same folder as this file)
from expression_parser import evaluate_expression

# Constants
IMAGE_TYPES = ("png", "jpg", "jpeg", "bmp")
MODEL_OPTIONS = list(cnn_backend.CNN_OPTIONS)  # ["CNN", "CNN + operators"]
EXPRESSION_MODEL = "CNN + operators"  # the option whose output is evaluated as arithmetic
PRETRAINED_DIR = "pretrained_models"
PRETRAINED_TYPES = (".keras", ".h5", ".pt", ".pth", ".pkl", ".joblib")
BG = "gainsboro"
HEADER_BG = "dim gray"


# GUI
class HNRSApp:
    def __init__(self, root):
        self.root = root
        root.title("Handwritten Number Recognition System")
        root.geometry("1150x760")
        root.minsize(980, 640)

        # States
        self.original_image = None
        self.processed_image = None
        self.digit_images = []
        self.boxes = []
        # Model name -> Trained/loaded model object
        self.models = {}  
        # Model name -> Pretrained file path (if loaded from disk)
        self.loaded_from = {}  
        self.history = {"loss": [], "acc": []}
        self.train_queue = queue.Queue()
        self.stop_event = threading.Event()
        # Keep PhotoImage references alive
        self._photos = {}  

        self._init_variables()
        self._build_layout()
        self._build_settings()
        self._build_notebook()
        self._build_output()
        self._set_status("Load an image to get started.")

    # Variables
    def _init_variables(self):
        self.model_var = tk.StringVar(value=MODEL_OPTIONS[0])
        # "train" or "pretrained"
        self.source_var = tk.StringVar(value="train")  
        self.weights_var = tk.StringVar(value="No pretrained model loaded.")
        self.epochs_var = tk.IntVar(value=10)
        self.batch_var = tk.IntVar(value=64)
        self.lr_var = tk.StringVar(value="0.001")
        self.arch_var = tk.StringVar(value=cnn_backend.CNN_ARCHS[0])
        self.aug_var = tk.StringVar(value="light")
        self.status_var = tk.StringVar()

    # Layout
    def _build_layout(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Settings
        self.settings = tk.Frame(self.root, width=250, relief="solid", borderwidth=2, background=BG)
        self.settings.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)
        self.settings.grid_propagate(False)

        # Center
        self.center = tk.Frame(self.root)
        self.center.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        # Output
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

    # Settings
    def _build_settings(self):
        tk.Label(self.settings, text="Settings", font=("Arial", 14), anchor="w", background=HEADER_BG).pack(fill="x")

        # Model + Hyper-parameters
        box = self._section(self.settings, "Model")
        combo = self._row(box, "Model:", lambda p: ttk.Combobox(p, textvariable=self.model_var,
                                                                 values=MODEL_OPTIONS, state="readonly", width=14))
        combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # Train new or Pretrained
        src = tk.Frame(box, background=BG)
        src.pack(fill="x", padx=4)
        for text, value in (("Train new", "train"), ("Pretrained", "pretrained")):
            tk.Radiobutton(src, text=text, value=value, variable=self.source_var,
                           command=self._on_source_change, background=BG).pack(side="left", padx=4)

        # Train new and Pretrained menus share the same section
        self.model_stack = tk.Frame(box, background=BG)
        self.model_stack.pack(fill="x")

        # "Train new": Hyper-parameters + Train controls
        self.train_frame = tk.Frame(self.model_stack, background=BG)
        self._row(self.train_frame, "Epochs:", lambda p: ttk.Spinbox(p, from_=1, to=200,
                                                                     textvariable=self.epochs_var, width=8))
        self._row(self.train_frame, "Batch size:", lambda p: ttk.Spinbox(p, from_=8, to=1024, increment=8,
                                                                         textvariable=self.batch_var, width=8))
        self._row(self.train_frame, "Learn rate:", lambda p: ttk.Entry(p, textvariable=self.lr_var, width=10))
        self._row(self.train_frame, "CNN arch:", lambda p: ttk.Combobox(p, textvariable=self.arch_var,
                  values=cnn_backend.CNN_ARCHS, state="readonly", width=10))
        self._row(self.train_frame, "Augment:", lambda p: ttk.Combobox(p, textvariable=self.aug_var,
                  values=cnn_backend.AUGMENTATIONS, state="readonly", width=10))

        btns = tk.Frame(self.train_frame, background=BG)
        btns.pack(fill="x", padx=4, pady=4)
        self.train_btn = ttk.Button(btns, text="Train", command=self.start_training)
        self.train_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_training, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))
        self.progress = ttk.Progressbar(self.train_frame, mode="determinate")
        self.progress.pack(fill="x", padx=6, pady=(0, 6))

        # "Pretrained": just a status line; the model loads automatically
        self.pre_frame = tk.Frame(self.model_stack, background=BG)
        tk.Label(self.pre_frame, textvariable=self.weights_var, wraplength=210, justify="left",
                 anchor="w", background=BG).pack(fill="x", padx=6, pady=6)

        self.train_frame.pack(fill="x")  # default source: train new

        ttk.Button(self.settings, text="Recognise Number", command=self.recognise).pack(
            fill="x", padx=12, pady=10, ipady=4
        )

    # Notebook
    def _build_notebook(self):
        self.nb = ttk.Notebook(self.center)
        self.nb.pack(fill="both", expand=True)

        # Input Tab
        self.tab_input = tk.Frame(self.nb)
        self.nb.add(self.tab_input, text="Input")

        bar = tk.Frame(self.tab_input)
        bar.pack(fill="x", pady=5, padx=5)
        ttk.Button(bar, text="Load image...", command=self.select_image).pack(side="left", padx=2)
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

        # Preprocessed Tab
        self.tab_pre = tk.Frame(self.nb)
        self.nb.add(self.tab_pre, text="Preprocessed")
        self.pre_label = tk.Label(self.tab_pre, text="Run recognition to see the preprocessed image.")
        self.pre_label.pack(expand=True, fill="both")

        # Segments Tab
        self.tab_seg = tk.Frame(self.nb)
        self.nb.add(self.tab_seg, text="Segments")
        self.seg_label = tk.Label(self.tab_seg, text="Run recognition to see the segmented digits.")
        self.seg_label.pack(fill="both", expand=True)
        self.thumb_frame = tk.Frame(self.tab_seg)
        self.thumb_frame.pack(fill="x", pady=8)

        # Training Tab
        self.tab_train = tk.Frame(self.nb)
        self.nb.add(self.tab_train, text="Training")
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax_loss = self.fig.add_subplot(121)
        self.ax_acc = self.fig.add_subplot(122)
        self._redraw_curves()
        self.curve_canvas = FigureCanvasTkAgg(self.fig, master=self.tab_train)
        self.curve_canvas.get_tk_widget().pack(fill="both", expand=True)

    # Output
    def _build_output(self):
        tk.Label(self.output, text="Output", font=("Arial", 14), anchor="w", background=HEADER_BG).pack(fill="x")
        self.result_var = tk.StringVar(value="-")
        row = tk.Frame(self.output, background=BG)
        row.pack(fill="both", expand=True, padx=10)
        tk.Label(row, text="Recognised number:", font=("Arial", 12), background=BG).pack(side="left")
        tk.Label(row, textvariable=self.result_var, font=("Arial", 28, "bold"), background=BG).pack(side="left", padx=15)
        self.detail_var = tk.StringVar()
        tk.Label(row, textvariable=self.detail_var, justify="left", anchor="w", background=BG).pack(side="left", padx=20)

    # Image input
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

    # Recognition
    def recognise(self):
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return

        model_name = self.model_var.get()

        if model_name not in self.models:
            if self.source_var.get() == "pretrained":
                messagebox.showinfo("No model loaded", f"No pretrained {model_name} model was found in "
                                    f"'{PRETRAINED_DIR}/'. Add one there, or switch to "
                                    "Train new.")
            else:
                messagebox.showinfo("Model not trained", f"Train the {model_name} model first, "
                                    "or switch to Pretrained.")
            return

        try:
            out = cnn_backend.recognise_cnn(self.models[model_name], self.original_image)
        except Exception as e:
            messagebox.showerror("Recognition failed", str(e))
            return

        # Preprocessed image (project preprocessing)
        self.processed_image = out["binary"]
        self._show(self.pre_label, "pre", self.processed_image, (700, 380))

        # Segmented symbols (project segmentation)
        self.digit_images, self.boxes = out["thumbs"], out["boxes"]
        if not out["results"]:
            self.result_var.set("?")
            self.detail_var.set("Nothing found in the image.")
            return
        self._show_segments(out["results"])

        # Recognised text
        self.result_var.set(out["text"])
        detail = ("Confidence per symbol:\n" +
                  "  ".join(f"{l}: {c:.0%}" for l, c in out["results"]))
        status = f"Recognised {len(out['results'])} symbol(s) using {model_name}."

        # "CNN + operators": evaluate the recognised text as an arithmetic expression
        if model_name == EXPRESSION_MODEL:
            value, err = evaluate_expression(out["text"])
            if err is None:
                detail = f"Result: {out['text']} = {value}\n" + detail
                status = f"{out['text']} = {value}  |  " + status
            else:
                # usually a misread symbol: point the user at the Segments tab
                detail = f"Cannot evaluate: {err}\n(check the Segments tab for misread symbols)\n" + detail
                status = f"Could not evaluate '{out['text']}': {err}"
        self.detail_var.set(detail)
        self._set_status(status)

    def _show_segments(self, results):
        # Annotated copy of the processed image with boxes
        annotated = self.processed_image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for (x0, y0, x1, y1), (lbl, _) in zip(self.boxes, results):
            draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
        self._show(self.seg_label, "seg", annotated, (700, 300))

        # One thumbnail + Prediction per digit
        for child in self.thumb_frame.winfo_children():
            child.destroy()

        for i, (digit, (lbl, conf)) in enumerate(zip(self.digit_images, results)):
            cell = tk.Frame(self.thumb_frame, relief="groove", borderwidth=1)
            cell.pack(side="left", padx=4)
            photo = ImageTk.PhotoImage(digit.resize((56, 56), Image.NEAREST))
            self._photos[f"thumb{i}"] = photo
            tk.Label(cell, image=photo).pack()
            tk.Label(cell, text=f"{lbl} ({conf:.0%})").pack()

    # Pretrained models
    def _on_source_change(self):
        """Swap the Train-new panel and the Pretrained panel."""
        if self.source_var.get() == "train":
            self.pre_frame.pack_forget()
            self.train_frame.pack(fill="x")
        else:
            self.train_frame.pack_forget()
            self.pre_frame.pack(fill="x")
            self.autodetect_pretrained()

    def _on_model_change(self, _event=None):
        if self.source_var.get() == "pretrained":
            self.autodetect_pretrained()

    def autodetect_pretrained(self):
        """Called automatically when Pretrained is selected or the model changes.
        Uses a model already loaded for this name, else looks in PRETRAINED_DIR
        for a file whose name starts with the model name (e.g. cnn_mnist.keras),
        else falls back to the frozen checkpoint from the CNN backend."""
        name = self.model_var.get()
        if name in self.loaded_from:
            self.weights_var.set(f"Loaded: {os.path.basename(self.loaded_from[name])}")
            return

        key = name.lower().replace(" ", "_")

        if os.path.isdir(PRETRAINED_DIR):
            for f in sorted(os.listdir(PRETRAINED_DIR)):
                if f.lower().startswith(key) and f.lower().endswith(PRETRAINED_TYPES):
                    self.load_pretrained_file(os.path.join(PRETRAINED_DIR, f))
                    return

        if cnn_backend.default_checkpoint(name).is_file():
            # Nothing in pretrained_models/: use the frozen CNN
            self.weights_var.set("Loading... (first load takes a few seconds)")
            self.root.update_idletasks()
            self.load_pretrained_file(str(cnn_backend.default_checkpoint(name)))
            return

        self.weights_var.set(f"No pretrained {name} model found in '{PRETRAINED_DIR}/'.")

    def load_pretrained_file(self, path):
        name = self.model_var.get()
        try:
            self.models[name] = cnn_backend.load_cnn(name, path)
        except Exception as e:
            messagebox.showerror("Could not load model", f"{os.path.basename(path)}\n\n{e}")
            return

        self.loaded_from[name] = path
        self.weights_var.set(f"Loaded: {os.path.basename(path)}")
        self._set_status(f"Pretrained {name} model loaded from {os.path.basename(path)}.")

    # Training
    def start_training(self):
        try:
            params = {
                "epochs": int(self.epochs_var.get()),
                "batch_size": int(self.batch_var.get()),
                "learning_rate": float(self.lr_var.get()),
                "arch": self.arch_var.get(),
                "augmentation": self.aug_var.get(),
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

            try:
                model = cnn_backend.train_cnn(model_name, params, on_epoch, self.stop_event)
            except Exception as e:  # show the error instead of hanging
                self.train_queue.put(("error", model_name, e))
                return
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
                    self.loaded_from.pop(name, None)  # now a freshly trained model
                    finished = True
                elif msg[0] == "error":
                    _, name, err = msg
                    messagebox.showerror("Training failed", f"{name}: {err}")
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

    # Helpers
    def _show(self, label, key, img, max_size):
        """Display a PIL image in a label, scaled to fit, keeping aspect ratio."""
        img = img.copy()
        img.thumbnail(max_size)
        photo = ImageTk.PhotoImage(img)
        # Prevent garbage collection
        self._photos[key] = photo  
        label.config(image=photo, text="")

    def _set_status(self, text):
        self.status_var.set(text)

def main():
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    HNRSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()