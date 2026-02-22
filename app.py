import flet as ft
from flet import (
    Page, Column, Row, Container, Text, Image,
    ElevatedButton,
    ProgressBar, MainAxisAlignment,
    CrossAxisAlignment, border_radius, padding,
    FontWeight, TextAlign, BoxShadow, Offset, Alignment
)
import numpy as np
from PIL import Image as PILImage
import os
import threading
from tkinter import filedialog
import tkinter as tk

# ── Try to import TensorFlow ──────────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_FILE      = "traffic_sign_model.h5"
CLASS_FILE      = "class_names.txt"
IMAGE_SIZE      = (64, 64)


def load_model_and_classes():
    """Loads the saved model and class names."""
    if not TF_AVAILABLE:
        return None, None, "TensorFlow not installed. Run: pip install tensorflow"
    if not os.path.exists(MODEL_FILE):
        return None, None, f"Model file '{MODEL_FILE}' not found.\nPlease run train_model.py first!"
    if not os.path.exists(CLASS_FILE):
        return None, None, f"Class names file '{CLASS_FILE}' not found.\nPlease run train_model.py first!"

    model = tf.keras.models.load_model(MODEL_FILE)
    with open(CLASS_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    return model, class_names, None


def predict_image(model, class_names, image_path):
    """Runs the model on an image and returns the prediction."""
    img = PILImage.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    predictions = model.predict(img_array, verbose=0)[0]
    top_index = int(np.argmax(predictions))
    confidence = float(predictions[top_index]) * 100

    # Top 3 predictions
    top3_indices = np.argsort(predictions)[::-1][:3]
    top3 = [(class_names[i].replace("_", " ").title(), float(predictions[i]) * 100) for i in top3_indices]

    return class_names[top_index].replace("_", " ").title(), confidence, top3


# ── Color Palette ─────────────────────────────────────────────────────────────
BG_DARK    = "#0D0F14"
BG_CARD    = "#161923"
BG_CARD2   = "#1E2330"
ACCENT     = "#FF4B4B"        # traffic-red accent
ACCENT2    = "#FFB74B"        # amber
GREEN      = "#4BFFB0"
TEXT_WHITE = "#F0F4FF"
TEXT_GRAY  = "#8A93A8"
BORDER     = "#2A3040"


def main(page: Page):
    page.title = "Traffic Sign Recognizer"
    page.bgcolor = BG_DARK
    page.window_width = 700
    page.window_height = 820
    page.window_resizable = True

    # ── Load model ─────────────────────────────────────────────────────────────
    model, class_names, load_error = load_model_and_classes()

    # ── State refs ─────────────────────────────────────────────────────────────
    selected_image_path = [None]

    # ── UI Components ──────────────────────────────────────────────────────────

    # Model status banner
    status_text = Text(
        f"✅ Model loaded — {len(class_names)} sign categories" if not load_error else f"⚠️  {load_error}",
        size=12,
        color=GREEN if not load_error else ACCENT2,
        text_align=TextAlign.CENTER,
    )
    status_banner = Container(
        content=status_text,
        bgcolor=BG_CARD2,
        border_radius=border_radius.all(8),
        padding=padding.symmetric(horizontal=20, vertical=10),
        margin=padding.symmetric(horizontal=32),
    )

    # Image preview area
    preview_placeholder = Column([
        Text("📁", size=48),
        Text("Upload an image to begin", size=14, color=TEXT_GRAY),
        Text("Supports: JPG, PNG, BMP", size=11, color=BORDER),
    ], horizontal_alignment=CrossAxisAlignment.CENTER, spacing=6)

    image_display = Image("", visible=False, width=280, height=210, fit="contain")

    preview_container = Container(
        content=ft.Stack([
            Container(
                content=preview_placeholder,
                alignment=Alignment(0, 0),
            ),
            Container(
                content=image_display,
                alignment=Alignment(0, 0),
            ),
        ]),
        width=320,
        height=240,
        bgcolor=BG_CARD2,
        border_radius=border_radius.all(16),
        border=ft.border.all(2, BORDER),
        alignment=Alignment(0, 0),
        shadow=BoxShadow(spread_radius=0, blur_radius=20, color="#20000000", offset=Offset(0, 4)),
    )

    # Result area
    result_title = Text("", size=22, weight=FontWeight.BOLD, color=TEXT_WHITE, text_align=TextAlign.CENTER)
    result_confidence = Text("", size=14, color=TEXT_GRAY, text_align=TextAlign.CENTER)
    result_badge = Container(visible=False)
    top3_column = Column([], spacing=8)

    result_card = Container(
        content=Column([
            result_badge,
            result_title,
            result_confidence,
            top3_column,
        ], horizontal_alignment=CrossAxisAlignment.CENTER, spacing=12),
        bgcolor=BG_CARD,
        border_radius=border_radius.all(16),
        padding=padding.all(24),
        margin=padding.symmetric(horizontal=32),
        border=ft.border.all(1, BORDER),
        visible=False,
    )

    progress = ProgressBar(visible=False, color=ACCENT, bgcolor=BORDER)

    # Error message
    error_text = Text("", color=ACCENT, size=13, text_align=TextAlign.CENTER)

    # ── File Picker Dialog ────────────────────────────────────────────────────
    def handle_file_pick(file_path):
        if file_path:
            selected_image_path[0] = file_path

            # Show preview
            image_display.src = file_path
            image_display.visible = True
            preview_placeholder.visible = False
            result_card.visible = False
            error_text.value = ""
            analyze_btn.disabled = False
            page.update()

    def pick_image(_):
        """Open native file picker using tkinter"""
        def open_picker():
            # Create hidden root window
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            # Open file picker
            file_path = filedialog.askopenfilename(
                title="Select a traffic sign image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("JPG files", "*.jpg *.jpeg"),
                    ("PNG files", "*.png"),
                    ("BMP files", "*.bmp"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()

            if file_path:
                handle_file_pick(file_path)

        # Run in separate thread to avoid blocking UI
        thread = threading.Thread(target=open_picker)
        thread.start()

    upload_btn = ElevatedButton(
        "Choose Image",
        icon="upload_file",
        on_click=pick_image,
        style=ft.ButtonStyle(
            bgcolor=BG_CARD2,
            color=TEXT_WHITE,
            side=ft.BorderSide(1, BORDER),
            shape=ft.RoundedRectangleBorder(radius=10),
            padding=padding.symmetric(horizontal=24, vertical=14),
        ),
    )

    def analyze(_):
        if not selected_image_path[0] or not model:
            return

        # Show loading
        progress.visible = True
        analyze_btn.disabled = True
        result_card.visible = False
        error_text.value = ""
        page.update()

        try:
            sign_name, confidence, top3 = predict_image(model, class_names, selected_image_path[0])

            # Confidence color
            conf_color = GREEN if confidence >= 75 else ACCENT2 if confidence >= 50 else ACCENT

            # Main result
            result_badge.content = Container(
                content=Text(
                    f"{confidence:.1f}% Confidence",
                    size=11, weight=FontWeight.BOLD, color=BG_DARK
                ),
                bgcolor=conf_color,
                border_radius=border_radius.all(20),
                padding=padding.symmetric(horizontal=12, vertical=5),
            )
            result_badge.visible = True
            result_title.value = sign_name
            result_confidence.value = "Top prediction"

            # Top 3 breakdown
            top3_column.controls.clear()
            top3_column.controls.append(
                Text("All predictions:", size=12, color=TEXT_GRAY)
            )
            for i, (name, prob) in enumerate(top3):
                bar_color = ACCENT if i == 0 else BORDER
                top3_column.controls.append(
                    Container(
                        content=Column([
                            Row([
                                Text(name, size=12, color=TEXT_WHITE, expand=True),
                                Text(f"{prob:.1f}%", size=12, color=conf_color if i == 0 else TEXT_GRAY),
                            ]),
                            Container(
                                content=Container(
                                    width=max(4, int(prob / 100 * 260)),
                                    height=4,
                                    bgcolor=ACCENT if i == 0 else ACCENT2 if i == 1 else TEXT_GRAY,
                                    border_radius=border_radius.all(4),
                                ),
                                bgcolor=BG_CARD2,
                                border_radius=border_radius.all(4),
                                height=4,
                            )
                        ], spacing=4),
                        bgcolor=BG_CARD2,
                        border_radius=border_radius.all(8),
                        padding=padding.all(10),
                    )
                )

            result_card.visible = True

        except Exception as ex:
            error_text.value = f"Error analyzing image: {str(ex)}"

        finally:
            progress.visible = False
            analyze_btn.disabled = False
            page.update()

    analyze_btn = ElevatedButton(
        "Analyze Sign",
        icon="search",
        on_click=analyze,
        disabled=True,
        style=ft.ButtonStyle(
            bgcolor={ft.ControlState.DEFAULT: ACCENT, ft.ControlState.DISABLED: BG_CARD2},
            color={ft.ControlState.DEFAULT: "#FFFFFF", ft.ControlState.DISABLED: TEXT_GRAY},
            shape=ft.RoundedRectangleBorder(radius=10),
            padding=padding.symmetric(horizontal=28, vertical=14),
        ),
    )

    # ── Layout ─────────────────────────────────────────────────────────────────
    page.add(
        ft.SafeArea(
            content=Column(
                controls=[
                    Container(
                        content=Text("Traffic Sign Recognition", size=18, weight=FontWeight.BOLD, color=TEXT_WHITE),
                        padding=padding.only(left=32, top=20, bottom=4),
                    ),
                    status_banner,
                    ft.Divider(color=BORDER, height=1),
                    # Preview
                    Container(
                        content=preview_container,
                        alignment=Alignment(0, 0),
                        padding=padding.symmetric(vertical=4),
                    ),
                    # Buttons
                    Row(
                        [upload_btn, analyze_btn],
                        alignment=MainAxisAlignment.CENTER,
                        spacing=12,
                    ),
                    error_text,
                    progress,
                    result_card,
                ],
                spacing=16,
                scroll=ft.ScrollMode.AUTO,
            )
        )
    )


if __name__ == "__main__":
    ft.app(target=main)
