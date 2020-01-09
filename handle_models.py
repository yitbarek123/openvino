import cv2
import numpy as np

#preprocess input image

def preprocessing(input_image,height,width):
    image=np.copy(input_image)
    image=cv2.resize(image, (width,height))
    image=image.transpose((2,0,1))
    image=image.reshape(1,3,height,width)
    return image

def handle_car(output, input_shape):
    color=output['color'].flatten()
    car_type=output['type'].flatten()
    color_pred=np.argmax(color)
    type_pred=np.argmax(car_type)
    return color_pred, type_pred

    