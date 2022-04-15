import textwrap
from PIL import Image, ImageDraw, ImageFont


def draw_image_caption(pil_img, caption, image_width=800, font_size=14):
    font = ImageFont.truetype("./arial.ttf", font_size)

    # Estimate char height and width
    tmp_drawer = ImageDraw.Draw(pil_img, 'RGBA')
    x, y, xx, yy = tmp_drawer.textbbox((10,10), "ABCD", font=font)
    char_w = int((xx-x)/4)
    char_h = yy-y

    # Estimate caption region size
    wrapped_text = textwrap.wrap(caption, width = int(image_width/char_w))
    text = "\n".join(wrapped_text)
    x, y, xx, yy = tmp_drawer.multiline_textbbox((10,10), text, font=font)
    text_h = yy-y

    # Resize original image
    w,h = pil_img.size
    image_height = int(h * image_width/w) # fixed ratio
    pil_img = pil_img.resize((image_width, image_height), Image.BICUBIC)
    
    # Create background convas
    offset = text_h + 15
    pil_bg = Image.new('RGBA', (image_width, image_height+offset), (255,255,255,255))
    pil_bg.paste(pil_img, (0,0))

    # Draw caption with a small margin
    margin = 5
    drawer = ImageDraw.Draw(pil_bg, 'RGBA')
    drawer.text((margin, image_height+margin), text, font=font, fill="black")
    
    return pil_bg