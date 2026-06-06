import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision.models import ResNet34_Weights, resnet34


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "srodek_resnet34_unet_final.pth"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 576
THRESHOLD = 0.465
CLOSING_KERNEL_SIZE = 7
FILL_LARGEST_EXTERNAL_CONTOUR = True
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3)
OVERLAY_COLOR_RGB = np.array([255, 50, 50], dtype=np.float32)
OVERLAY_ALPHA = 0.45


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class ResNet34UNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        encoder = resnet34(weights=weights)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.dec4 = ConvBlock(512 + 256, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 64, 64)
        self.dec0 = ConvBlock(64, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        y = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec4(torch.cat([y, x3], dim=1))

        y = F.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, x2], dim=1))

        y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, x1], dim=1))

        y = F.interpolate(y, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([y, x0], dim=1))

        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        y = self.dec0(y)
        return self.final_conv(y)


def load_torch_checkpoint(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def read_image_rgb(path):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_for_model(image_rgb):
    return cv2.resize(image_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)


def image_to_tensor(image_rgb, device):
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGE_MEAN) / IMAGE_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0).float().to(device)


def apply_closing(mask, kernel_size):
    if kernel_size <= 0:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def fill_largest_external_contour(mask):
    mask_255 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask_255, dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 255, thickness=-1)
    return (filled > 0).astype(np.uint8)


def postprocess_mask(mask):
    processed = apply_closing(mask, CLOSING_KERNEL_SIZE)
    if FILL_LARGEST_EXTERNAL_CONTOUR:
        processed = fill_largest_external_contour(processed)
    return processed


def make_overlay(image_rgb, mask):
    overlay = image_rgb.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        (1.0 - OVERLAY_ALPHA) * overlay[mask_bool]
        + OVERLAY_ALPHA * OVERLAY_COLOR_RGB
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


class SrodekSegmenter:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, device="auto"):
        self.model_path = Path(model_path)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = ResNet34UNet(pretrained=False).to(self.device)
        checkpoint = load_torch_checkpoint(self.model_path, self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_rgb):
        resized = resize_for_model(image_rgb)
        tensor = image_to_tensor(resized, self.device)
        probability = torch.sigmoid(self.model(tensor)).squeeze().cpu().numpy()
        raw_mask = (probability > THRESHOLD).astype(np.uint8)
        mask = postprocess_mask(raw_mask)
        overlay = make_overlay(resized, mask)
        return mask, overlay, resized

    def predict_file(self, image_path):
        image_rgb = read_image_rgb(image_path)
        return self.predict(image_rgb)

    def save_overlay(self, image_path, output_path):
        _, overlay, _ = self.predict_file(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), overlay_bgr):
            raise ValueError(f"Cannot write output image: {output_path}")
        return output_path


def pil_image_for_gui(image_rgb, max_width, max_height):
    image = Image.fromarray(image_rgb)
    image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def run_gui(model_path=DEFAULT_MODEL_PATH, device="auto"):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    colors = {
        "bg": "#111318",
        "panel": "#1A1D24",
        "panel_alt": "#20242D",
        "border": "#303642",
        "text": "#F3F5F8",
        "muted": "#A8B0BD",
        "accent": "#E84D5B",
        "accent_hover": "#F06470",
        "button": "#2A303A",
        "button_hover": "#343B48",
        "disabled": "#555C68",
    }
    fonts = {
        "title": ("Segoe UI", 18, "bold"),
        "subtitle": ("Segoe UI", 10),
        "section": ("Segoe UI", 11, "bold"),
        "body": ("Segoe UI", 10),
        "button": ("Segoe UI", 10, "bold"),
    }

    root = tk.Tk()
    root.title("Srodek Segmenter")
    root.configure(bg=colors["bg"])
    root.minsize(980, 650)

    try:
        segmenter = SrodekSegmenter(model_path=model_path, device=device)
    except Exception as exc:
        messagebox.showerror("Srodek Segmenter", str(exc))
        root.destroy()
        return

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    panel_width = max(360, min(IMAGE_WIDTH, (screen_width - 180) // 2))
    panel_height = max(220, min(IMAGE_HEIGHT, screen_height - 250))

    root.overlay_image = None
    root.mask_image = None
    root.current_image_path = None

    app = tk.Frame(root, bg=colors["bg"], padx=22, pady=20)
    app.pack(fill="both", expand=True)
    app.grid_columnconfigure(0, weight=1)
    app.grid_columnconfigure(1, weight=1)
    app.grid_rowconfigure(1, weight=1)

    header = tk.Frame(app, bg=colors["bg"])
    header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
    header.grid_columnconfigure(0, weight=1)

    title_block = tk.Frame(header, bg=colors["bg"])
    title_block.grid(row=0, column=0, sticky="w")
    tk.Label(
        title_block,
        text="Srodek Segmenter",
        bg=colors["bg"],
        fg=colors["text"],
        font=fonts["title"],
    ).pack(anchor="w")
    tk.Label(
        title_block,
        text="Load a screenshot and preview the final mask overlay.",
        bg=colors["bg"],
        fg=colors["muted"],
        font=fonts["subtitle"],
    ).pack(anchor="w", pady=(3, 0))

    status_var = tk.StringVar(value="Ready")
    file_var = tk.StringVar(value="No image selected")
    status_label = tk.Label(
        header,
        textvariable=status_var,
        bg=colors["panel_alt"],
        fg=colors["text"],
        font=fonts["body"],
        padx=14,
        pady=8,
    )
    status_label.grid(row=0, column=1, sticky="e")

    def make_preview_card(parent, title, empty_text, column):
        card = tk.Frame(
            parent,
            bg=colors["panel"],
            highlightbackground=colors["border"],
            highlightthickness=1,
            padx=14,
            pady=14,
        )
        card.grid(row=1, column=column, sticky="nsew", padx=(0, 9) if column == 0 else (9, 0))
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)

        tk.Label(
            card,
            text=title,
            bg=colors["panel"],
            fg=colors["text"],
            font=fonts["section"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        preview = tk.Label(
            card,
            text=empty_text,
            bg=colors["panel_alt"],
            fg=colors["muted"],
            font=fonts["body"],
            width=54,
            height=22,
            relief="flat",
            bd=0,
        )
        preview.grid(row=1, column=0, sticky="nsew")
        return preview

    input_label = make_preview_card(app, "Input", "Open an image to begin", 0)
    overlay_label = make_preview_card(app, "Output", "The mask overlay will appear here", 1)

    footer = tk.Frame(app, bg=colors["bg"])
    footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(16, 0))
    footer.grid_columnconfigure(0, weight=1)

    file_label = tk.Label(
        footer,
        textvariable=file_var,
        bg=colors["bg"],
        fg=colors["muted"],
        font=fonts["body"],
        anchor="w",
    )
    file_label.grid(row=0, column=0, sticky="ew")

    button_bar = tk.Frame(footer, bg=colors["bg"])
    button_bar.grid(row=0, column=1, sticky="e")

    def make_button(parent, text, command, accent=False):
        bg = colors["accent"] if accent else colors["button"]
        hover = colors["accent_hover"] if accent else colors["button_hover"]
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=colors["text"],
            activebackground=hover,
            activeforeground=colors["text"],
            disabledforeground=colors["disabled"],
            font=fonts["button"],
            relief="flat",
            bd=0,
            padx=16,
            pady=9,
            cursor="hand2",
        )
        button.bind("<Enter>", lambda _event: button.configure(bg=hover) if button["state"] == "normal" else None)
        button.bind("<Leave>", lambda _event: button.configure(bg=bg) if button["state"] == "normal" else None)
        return button

    def load_image_dialog():
        image_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not image_path:
            return

        try:
            status_var.set("Processing...")
            root.update_idletasks()
            mask, overlay, resized = segmenter.predict_file(image_path)
        except Exception as exc:
            status_var.set("Ready")
            messagebox.showerror("Srodek Segmenter", str(exc))
            return

        input_photo = pil_image_for_gui(resized, panel_width, panel_height)
        overlay_photo = pil_image_for_gui(overlay, panel_width, panel_height)

        root.input_photo = input_photo
        root.overlay_photo = overlay_photo
        root.overlay_image = overlay
        root.mask_image = mask
        root.current_image_path = image_path

        input_label.configure(image=input_photo, text="", width=0, height=0)
        overlay_label.configure(image=overlay_photo, text="", width=0, height=0)
        root.title(f"Srodek Segmenter - {Path(image_path).name}")
        file_var.set(f"Loaded: {Path(image_path).name}")
        status_var.set("Ready")

    def save_overlay_dialog():
        if root.overlay_image is None:
            messagebox.showinfo("Srodek Segmenter", "Choose an image first.")
            return
        output_path = filedialog.asksaveasfilename(
            title="Save output",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not output_path:
            return
        overlay_bgr = cv2.cvtColor(root.overlay_image, cv2.COLOR_RGB2BGR)
        if cv2.imwrite(output_path, overlay_bgr):
            messagebox.showinfo("Srodek Segmenter", f"Saved: {output_path}")
        else:
            messagebox.showerror("Srodek Segmenter", f"Cannot save: {output_path}")

    open_button = make_button(button_bar, "Open Image", load_image_dialog, accent=True)
    open_button.pack(side="left", padx=(0, 8))
    save_button = make_button(button_bar, "Save Output", save_overlay_dialog)
    save_button.pack(side="left", padx=(0, 8))
    close_button = make_button(button_bar, "Close", root.destroy)
    close_button.pack(side="left")

    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Segment srodek on an image.")
    parser.add_argument("--gui", action="store_true", help="open an interactive image picker")
    parser.add_argument("--input", type=Path, help="input image path")
    parser.add_argument("--output", type=Path, help="output overlay PNG path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="model checkpoint path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gui or args.input is None:
        run_gui(model_path=args.model, device=args.device)
        return

    if args.output is None:
        raise SystemExit("--output is required when --input is used")

    segmenter = SrodekSegmenter(model_path=args.model, device=args.device)
    output_path = segmenter.save_overlay(args.input, args.output)
    print(f"Saved output: {output_path}")


if __name__ == "__main__":
    main()
