import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import torch.onnx
import onnx
import onnxruntime
import numpy as np

# parser = argparse.ArgumentParser(description='for face verification')
# parser.add_argument("-s", "--save", help="whether save",action="store_true")
# parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
# parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
# parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
# parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
# args = parser.parse_args()

def rescale_video_img(frame1,scale=0.75):
    width = int(frame1.shape[1] * scale)
    height = int(frame1.shape[0] * scale)
    dim = (width,height)
    return (cv2.resize(frame1,dim,interpolation=cv2.INTER_AREA))

def main(args):

    conf = get_config(False)
    # mtcnn = MTCNN()
    print('arcface loaded')

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    batch_size = 1
    x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)
    torch_out = learner.model(x)
    
    # print(learner.model)
   
    # torch.onnx.export(learner.model,               # model being run
    #                 x,                         # model input (or a tuple for multiple inputs)
    #                 "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
    #                 export_params=True,        # store the trained parameter weights inside the model file
    #                 opset_version=10,          # the ONNX version to export the model to
    #                 do_constant_folding=True)

    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx_model)



    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


    



if __name__ == "__main__":
    args = lambda : None
    args.width = 800
    args.height = 800
    args.image = None
    args.save = True
    args.threshold = 1.54
    args.update = False
    args.tta = True
    args.score = False
    args.video_speedup = False
    main(args)         
    
    

   


