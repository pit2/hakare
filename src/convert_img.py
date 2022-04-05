from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

image = Image.open("data/images/ETL1/0x003d/059245.png")

image.show()

transform = transforms.Compose([transforms.PILToTensor()])

img_tensor = transform(image)

img_tensor = img_tensor / 255

print(img_tensor)

save_image(img_tensor, "data/tmp/back.png")
