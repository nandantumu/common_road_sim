from PIL import Image

# Open the input image
input_image_path = '../resource/clark_park/ClarkPark.jpg'
output_image_path = '../resource/clark_park/clark_park_greyscale.pgm'

# Open the image file
with Image.open(input_image_path) as img:
    # Convert the image to grayscale
    grayscale_img = img.convert('L')
    # Save the grayscale image as PNG
    grayscale_img.save(output_image_path, format='PPM')

print(f"Converted {input_image_path} to grayscale and saved as {output_image_path}")