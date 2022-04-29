
# Cummon Libraries
import os
from matplotlib import font_manager

# Custom Libraries

# Sources

def get_font_paths(font_dir_path, font_name):
    """
    get font path form font_dir_path

    :param font_dir_path: font directory path ex) "/Users/clmn/library/Fonts"
    :param font_name: font name ex) Amarillo
    return: target_font_path, font_file_name
    """
    font_list = font_manager.findSystemFonts(fontpaths=font_dir_path)
    target_font_path = ""
    for font in font_list:
        if font_name in font:
            target_font_path = font

    font_file_name = target_font_path.split(os.sep)[-1]

    return target_font_path, font_file_name

if __name__ == "__main__":
    """
    You can see default font using the source.
    
    import matplotlib
    # matplotlib.font_manager.fontManager.ttflist
    """

    """
    You can see font as image file 
    
    import matplotlib
    from PIL import Image, ImageFont, ImageDraw
    import os
    
    target_list = []
    for f in matplotlib.font_manager.fontManager.ttflist:
        if f.style == "italic":
            print(f.name, f.fname, f.style, f.size, f.variant)
    
            title_font = ImageFont.truetype(f.fname, 100)
            title_text = "a,d,e,g,m,o,p,s,u,z"
            my_image = Image.new('RGBA', (2000, 400), (0, 0, 0, 0))
            image_editable = ImageDraw.Draw(my_image)
    
            image_editable.text((15, 15), title_text, (237, 230, 211), font=title_font)
    
            my_image.save(os.path.join("/Users/clmn/Desktop/font_list", f.name + ".png"))
    
            target_list.append(f)
    """
    get_font_paths("/Users/clmn/library/Fonts", "Amarillo")

