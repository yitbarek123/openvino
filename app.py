import argparse
import cv2
import numpy as np

from handle_models import handle_output,preprocessing

from inference import Network

CAR_COLORS=["white","gray","yellow","red","green","blue","black"]
CAR_TYPES=["car","bus","truck","van"]

def get_args():
    "get argument from the command line"

    parser=argparse.ArgumentParser("Basic Edge App with Inference Engine")
    c_desc="CPU extension file location, if applicable"
    d_desc="Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc="location of the input image"
    m_desc="The location of the model XML"
    t_desc="The type of model: POSE, TEXT or CAR_META"

    # classify arguments as optional and required groups
    parser._action_groups_pop.()
    required=parser.add_argument_group('required arguments')
    optional=parser.add _argument_group('optional arguments')

    #-- create the arguments

    required.add_argument("-i",help=i_desc,required=True)
    required.add_argument("-m",help=m_desc,required=True)
    required.add_argument("-t",help=t_desc,required=True)
    optional.add_argument("-c",help=c_desc,default=None)
    optional.add_argument("-d",help=d_desc,default="CPU")
    args=parser.parse_args()
    return args

def get_mask(processed_output):
    "given an image size processed output for a semantic mask"
    empty=np.zeros(processed_output.shape)
    mask=np.dstack((empty,processed_output,empty))
    return mask

def create_output_image(model_type,image,output):
    if model_type=="CAR_META":
        #get the color and car type from thier lists
        color=CAR_COLORS[output[0]]
        car_type=CAR_TYPES[output[1]]
        
        #Scale the output text by the image shape
        scaler=max(int(image.shape[0]/1000),1)
        image=cv2.putText(image,
            "Color: {}, Type: {}".format(color, car_type),
            (50*scaler,100*scaler),cv2.FONT_HERSHEY_SIMPLEX,
            2*scaler,(255,255,255),3*scaler)
        return image
    else:
        print("Unknown model type, unable to create output image.")
        return image

def perform_inference(args):
    #create a network for using the inference engine
    inference_network=Network()
    #load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)
    #read the input the image
    image=cv2.imread(args.i)

    #preprocess the input image
    preprocessed_image=preprocessing(image,h,w)
    
    #perform infernce on the image
    inference_network.sync_inference(preprocessed_image)

    #obtain the output of the inference request
    output=inference_network.extract_output()

    output_func=handle_output(args.t)
    processed_output=output_func(output,image.shape)

    #create an output image based on network

    output_image=create_output_image(args.t,image,preprocessed_output)
    cv2.imwrite("outputs/{}-output.png".format(args.t),output_image)

def main():
    args=get_args()
    perform_inference(args)

if __name__=="__main__":
    main()

