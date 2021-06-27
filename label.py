import os
import PIL
from PIL import ImageChops, ImageOps, Image


threshold = 10
base_dir = "/media/MX500/CS_BA_Data/render_output/"
base_img_path = os.path.join(base_dir, "render_full.png")
if not os.path.isfile(base_img_path):
    quit()
base = Image.open(base_img_path)
rgb_base = base.convert('RGB')
i = 0
img_path = os.path.join(base_dir, "render_img_" + str(i) + ".png")
while os.path.isfile(img_path):
  rgb_img = Image.open(img_path).convert('RGB')
  diff = ImageChops.difference(rgb_base,rgb_img)
  # TODO: Filter image , just have 0/255 as output
  diff = diff.point(lambda p: p > threshold and 255)
  diff = ImageOps.grayscale(diff)
  #diff.show()
  diff.save(os.path.join(base_dir, "diff_" + str(i) + ".png"), "PNG")
  i += 1
  img_path = os.path.join(base_dir, "render_img_" + str(i) + ".png")
