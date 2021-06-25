import cv2
from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import time
import torch.onnx
import onnx

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore


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

def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized

def main(args):

    conf = get_config(False)

    mtcnn = MTCNN()
    print('arcface loaded')

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    
    # if conf.device.type == 'cpu':
    #     learner.load_state(conf, 'cpu_final.pth', True, True)
    # else:
    #     learner.load_state(conf, 'final.pth', True, True)
    # learner.model.eval()


    # batch_size = 1    # just a random number
    # x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)
    # torch_out = learner.model(x)

    # torch.onnx.export(learner.model,               # model being run
    #               x,                         # model input (or a tuple for multiple inputs)
    #               "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
    #               export_params=True,        # store the trained parameter weights inside the model file
    #               opset_version=10,          # the ONNX version to export the model to
    #               do_constant_folding=True)
    
    # onnx_model = onnx.load("super_resolution.onnx")
    # onnx.checker.check_model(onnx_model)

    # print('learner loaded')
#------------------------------------------------------------------------------------------------------------------------------
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    # if args.cpu_extension and 'CPU' in args.device:
    #     ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    # if "CPU" in args.device:
    #     supported_layers = ie.query_network(net, "CPU")
    #     not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    #     if len(not_supported_layers) != 0:
    #         log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
    #                   format(args.device, ', '.join(not_supported_layers)))
    #         log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
    #                   "or --cpu_extension command line argument")
    #         sys.exit(1)

    # assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    # assert len(net.outputs) == 1, "Sample supports only single output topologies"

    # log.info("Preparing input blobs")
    # input_blob = next(iter(net.inputs))
    # out_blob = next(iter(net.outputs))
    # net.batch_size = len(args.input)

    # Read and pre-process input images
    # n, c, h, w = net.inputs[input_blob].shape
    # images = np.ndarray(shape=(n, c, h, w))
    # for i in range(n):
    #     image = cv2.imread(args.input[i])
    #     if image.shape[:-1] != (h, w):
    #         log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
    #         image = cv2.resize(image, (w, h))
    #     image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    #     images[i] = image
    # log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

#------------------------------------------------------------------------------------------------------------------------------

    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # inital camera
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap = cv2.VideoCapture(r"D:\face_recog\Face-Recognition_paul\videos\test3.mp4")
    cap.set(3,500)
    cap.set(4,500)    
    # video_capture = cv2.VideoCapture("./videos/test3.mp4")
    # widthh = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # heightt = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    # video_recording = cv2.VideoWriter('output.avi', fourcc, 10,(int(widthh), int(heightt)))
    _, frame = cap.read()
    frame = resize_image(frame, 0.5)
    out = cv2.VideoWriter(r'D:\face_recog\Face-Recognition_paul\output_video\output3.avi',cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (frame.shape[1], frame.shape[0]), isColor=True)
    
    total_frames_passed = 0

    while cap.isOpened():
        
        isSuccess,frame = cap.read()
        
        if args.video_speedup:               
            total_frames_passed += 1
            if total_frames_passed % args.video_speedup != 0:
                continue
    
        if isSuccess:            
            try:
                # image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                start_time = time.time()
                frame = resize_image(frame, 0.5)
                image = Image.fromarray(frame)
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice

                # Start sync inference
                log.info("Starting inference in synchronous mode")
                for img in faces:  
                    res = exec_net.infer(inputs={conf.test_transform(img).unsqueeze(0)})
                    embs.append(res)
                    
                source_embs = torch.cat(embs)
                diff = source_embs.unsqueeze(-1) - targets.transpose(1,0).unsqueeze(0)
                dist = torch.sum(torch.pow(diff, 2), dim=1)
                score, results = torch.min(dist, dim=1)
                results[score > conf.threshold] = -1 # if no match, set idx to -1   

                # # results, score = learner.infer(conf, faces, targets, args.tta)
                # embs = []
                # for img in faces:
                #     if tta:
                #         mirror = trans.functional.hflip(img)
                #         emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                #         emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                #         embs.append(l2_norm(emb + emb_mirror))
                #     else:                        
                #         embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
                # source_embs = torch.cat(embs)
                
                # diff = source_embs.unsqueeze(-1) - targets.transpose(1,0).unsqueeze(0)
                # dist = torch.sum(torch.pow(diff, 2), dim=1)
                # score, results = torch.min(dist, dim=1)
                # results[score > self.threshold] = -1 # if no match, set idx to -1
                
                # print(score[0])

                FPS = 1.0 / (time.time() - start_time)

                frame = cv2.putText(frame,
                        'FPS: {:.1f}'.format(FPS),(10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,
                        (0,255,0),
                        2,
                        cv2.LINE_AA)

                for idx,bbox in enumerate(bboxes):
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        if float('{:.2f}'.format(score[idx])) > .99:
                            # name = names[results[idx]+1]
                            name = names[0]
                            print("if score {:.3f}".format(score[idx]))
                            print("if name %s" %name)
                        else:
                            # name = names[0]    
                            name = names[results[idx]+1]
                            print("else score {:.3f}".format(score[idx]))
                            print("else name %s" %name)
                        # frame1 = rescale_video_img(frame)
                frame = draw_box_name(bbox, name, frame)
                
                        


                # frame = cv2.cvtColor(np.asarray(frame))
                
                # video_recording.write(frame)
            except:
                pass 
            out.write(frame) 
            cv2.imshow('video', frame) 
            # ret, jpeg = cv2.imencode('.jpg', frame)
            # return jpeg.tostring()
            # frame = resize_image(frame,0.5)
            
                        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                
    cap.release()
    out.release()
    # video_recording.release()
    cv2.destroyAllWindows()



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
    args.device = 'cpu'
    main(args)         
    
    

   


