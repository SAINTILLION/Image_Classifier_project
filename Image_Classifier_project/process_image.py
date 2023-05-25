from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    image = Image.open(image)  # open image
    
    if image.size[0] > image.size[1]: # If width is greater than the height
        image.thumbnail((10000, 256)) # Adjust the shortest height to 256pixels
        
    else:
        image.thumbnail((256, 1000))
        
    # Crop the image
    width, height = image.size
    
    left = (width - 224) / 2
    bottom = (height - 224) / 2
    right = (left + 224)
    top = (bottom + 224)
    
    image = image.crop(box =(left, bottom, right, top))
    
    # Normalize the image
    image = np.array(image) / 255
    
    means = [0.485, 0.456, 0.406] # Mean values
    std = [0.229, 0.224, 0.225] # standard deviation values
    
    image = (image - means) / std
    
    # Move the color channel to the first dimension
    image = image.transpose(2, 0, 1)
    return image
    
    
    
    
    
    
    
    
    
    