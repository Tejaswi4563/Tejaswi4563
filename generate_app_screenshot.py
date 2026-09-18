from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 800
BG = (44, 62, 80)
PANEL = (52, 73, 94)
LIGHT_PANEL = (69, 90, 109)
BLUE = (52, 152, 219)
RED = (231, 76, 60)
GREEN = (39, 174, 96)
ORANGE = (243, 156, 18)
PURPLE = (155, 89, 182)
TEXT = (255, 255, 255)
MUTED = (221, 221, 221)
YELLOW = (255, 214, 102)


def load_font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size) if bold else ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_button(draw, x, y, w, h, text, color, font):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=color)
    text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
    tx = x + (w - text_w) / 2
    ty = y + (h - text_h) / 2 - 2
    draw.text((tx, ty), text, fill=TEXT, font=font)


def draw_panel(draw, x, y, w, h, fill):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=fill)


def main():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # Title
    title_font = load_font(32, bold=True)
    draw.text((60, 30), "AI Background & Music Remover", fill=TEXT, font=title_font)

    # Status panel
    status_y = 90
    status_panel = (50, status_y, 1230, 170)
    draw.rounded_rectangle(status_panel, radius=12, fill=PANEL)

    status_font = load_font(18)
    status_lines = [
        "Available modules:",
        "✅ AI Background Removal",
        "✅ Audio Processing",
        "✅ Basic Image Processing",
    ]
    for idx, line in enumerate(status_lines):
        y = 112 + idx * 22
        color = YELLOW if idx == 0 else TEXT
        draw.text((80, y), line, fill=color, font=status_font)

    # Notebook area
    notebook_x, notebook_y, notebook_w, notebook_h = 50, 200, 1180, 560
    draw.rounded_rectangle((notebook_x, notebook_y, notebook_x + notebook_w, notebook_y + notebook_h), radius=12, fill=PANEL)

    # Tab bar
    tab_y = notebook_y + 10
    draw.rounded_rectangle((notebook_x + 18, tab_y, notebook_x + 420, tab_y + 42), radius=10, fill=LIGHT_PANEL)
    draw.rounded_rectangle((notebook_x + 440, tab_y, notebook_x + 770, tab_y + 42), radius=10, fill=LIGHT_PANEL)

    tab_font = load_font(15, bold=True)
    draw.text((notebook_x + 48, tab_y + 11), "Image Background Removal", fill=TEXT, font=tab_font)
    draw.text((notebook_x + 470, tab_y + 11), "Audio Noise Removal", fill=TEXT, font=tab_font)

    # Image controls row
    control_y = notebook_y + 70
    buttons = [
        ("Upload Image", BLUE, 80),
        ("AI Remove Background", RED, 240),
        ("Basic Remove Background", PURPLE, 520),
        ("Replace Background", ORANGE, 790),
        ("Save Image", GREEN, 1035),
    ]
    button_font = load_font(13, bold=True)
    for label, color, x in buttons:
        draw_button(draw, notebook_x + x, control_y, 150, 36, label, color, button_font)

    # Left & right image panels
    left_x = notebook_x + 35
    right_x = notebook_x + 610
    img_y = notebook_y + 130
    panel_w = 500
    panel_h = 330

    draw_panel(draw, left_x, img_y, panel_w, panel_h, (44, 62, 80))
    draw_panel(draw, right_x, img_y, panel_w, panel_h, (44, 62, 80))

    # Simulated image previews
    preview_font = load_font(18, bold=True)
    draw.text((left_x + 180, img_y + 10), "Original Image", fill=TEXT, font=preview_font)
    draw.text((right_x + 190, img_y + 10), "Processed Image", fill=TEXT, font=preview_font)

    # Simulated face/person artwork on left
    person_x = left_x + 140
    person_y = img_y + 120
    draw.ellipse((person_x, person_y, person_x + 180, person_y + 180), fill=(240, 196, 139))
    draw.rounded_rectangle((person_x + 40, person_y + 170, person_x + 140, person_y + 250), radius=20, fill=(52, 152, 219))
    draw.rectangle((person_x + 20, person_y + 125, person_x + 80, person_y + 180), fill=(61, 90, 128))
    draw.rectangle((person_x + 120, person_y + 125, person_x + 180, person_y + 180), fill=(61, 90, 128))
    draw.ellipse((person_x + 55, person_y + 60, person_x + 95, person_y + 100), fill=(21, 32, 44))
    draw.ellipse((person_x + 110, person_y + 60, person_x + 150, person_y + 100), fill=(21, 32, 44))
    draw.arc((person_x + 60, person_y + 90, person_x + 140, person_y + 145), start=200, end=340, fill=(0, 0, 0), width=4)

    # Simulated processed result with background removed
    bgcolor = (255, 255, 255)
    for yy in range(img_y + 60, img_y + 330):
        for xx in range(right_x + 30, right_x + 470):
            if ((xx - (right_x + 235)) ** 2 + (yy - (img_y + 165)) ** 2) < 10000:
                pass
    draw.ellipse((right_x + 155, img_y + 120, right_x + 330, img_y + 290), fill=(240, 196, 139))
    draw.rounded_rectangle((right_x + 185, img_y + 250, right_x + 300, img_y + 330), radius=20, fill=(52, 152, 219))
    draw.rectangle((right_x + 170, img_y + 150, right_x + 220, img_y + 210), fill=(61, 90, 128))
    draw.rectangle((right_x + 265, img_y + 150, right_x + 315, img_y + 210), fill=(61, 90, 128))
    draw.ellipse((right_x + 190, img_y + 90, right_x + 225, img_y + 125), fill=(21, 32, 44))
    draw.ellipse((right_x + 270, img_y + 90, right_x + 305, img_y + 125), fill=(21, 32, 44))
    draw.arc((right_x + 200, img_y + 120, right_x + 285, img_y + 175), start=200, end=340, fill=(0, 0, 0), width=4)

    # Footer-like audio panel label at bottom
    bottom_y = notebook_y + 490
    draw.rounded_rectangle((notebook_x + 35, bottom_y, notebook_x + 1145, bottom_y + 40), radius=8, fill=(44, 62, 80))
    draw.text((notebook_x + 60, bottom_y + 10), "Status: Background removal ready to use", fill=TEXT, font=load_font(16))

    img.save("app_screenshot.png")
    print("Saved app_screenshot.png")


if __name__ == "__main__":
    main()
