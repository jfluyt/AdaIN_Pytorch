import argparse
import cv2
import numpy as np

from train import train_network
from evaluate import evaluate_network

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-no', type=int,
            help='cpu: -1, gpu: 0 ~ n', default=0)

    parser.add_argument('--train', action='store_true',
            help='train flag', default=False)

    parser.add_argument('--video', type=str, 
            help='test content video', default=None)

    parser.add_argument('--style', type=str, nargs='+',
            help='test style image', default=None)

    parser.add_argument('--content-dir', type=str,
            help='train content dir', default=None)

    parser.add_argument('--style-dir', type=str,
            help='train style dir', default=None)

    parser.add_argument('--layers', type=int, nargs='+',
            help='layer indices', default=[1, 6, 11, 20])

    parser.add_argument('--style-strength', type=float,
            help='content-style strength interpolation factor, 1: style, 0: content', default=1.0)


    parser.add_argument('--imsize', type=int,
            help='size to resize image', default=512)

    parser.add_argument('--cropsize', type=int,
            help='size to crop image', default=None)

    parser.add_argument('--cencrop', action='store_true',
            help='crop the center region of the image', default=False)

    parser.add_argument('--lr', type=float,
            help='learning rate', default=1e-4)

    parser.add_argument('--max-iter', type=int,
            help='number of iterations to train the network', default=80000)

    parser.add_argument('--batch-size', type=int,
            help='batch size', default=16)

    parser.add_argument('--style-weight', type=float,
            help='style loss weight', default=100)

    parser.add_argument('--check-iter', type=int,
            help='number of iteration to check train logs', default=500)

    parser.add_argument('--load-path', type=str,
            help='model load path', default=None)

    parser.add_argument('--spatial-control-video',  action='store_true',
            help='use of spatial control in video', default=None)

    
    parser.add_argument('--preserve-color', action='store_true',
            help='flag for color preserved stylization', default=None)

    parser.add_argument('--interpolation-weights', type=float, nargs='+',
            help='multi-style interpolation weights', default=None)   
             
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    if args.video :
        cap = cv2.VideoCapture(args.video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 1 / length
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height =int(cap.get( cv2.CAP_PROP_FRAME_HEIGHT)) 
        out = cv2.VideoWriter('output_test.avi', fourcc, fps, (frame_width, frame_height))
        if args.spatial_control_video :
            ret, firstFrame = cap.read()
            if ret is False :
                raise Exception('not possible')

            tracker = cv2.TrackerCSRT_create()
            bbox = (287, 23, 86, 320)
            bbox = cv2.selectROI(firstFrame)
            ok = tracker.init(firstFrame, bbox)
            while True :
                ret, frame = cap.read()
                if(ok == False) :
                    break
                cv2.imwrite('frame.jpg', frame)
                args.content = 'frame.jpg'

                height, width = frame.shape[:2]
                mark1 = np.zeros((height, width), np.uint8)
                mark2 = 255 * np.ones((height, width), np.uint8)
                ok, bbox = tracker.update(frame)
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(mark1, p1, p2, (255,0,0), -1)
                    cv2.rectangle(mark2, p1, p2, (0,0,0), -1)

                    cv2.imwrite('mask1.jpg', mark1)
                    cv2.imwrite('mask2.jpg', mark2)
                    args.mask = ['mask1.jpg', 'mask2.jpg']
                evaluate_network(args)
                stylised_frame = cv2.imread('stylized_image.jpg')
                stylised_frame = cv2.resize(stylised_frame, (frame_width, frame_height), interpolation = cv2.INTER_AREA) 
                out.write(stylised_frame)
        else :
            ''' 
                uncomment for transition of two styles. Can also be updated for more than two styles 
            '''
            #countr = 1
            #countr2 = 0
            while True:
                #countr -= i
                #countr = float(countr)
                #countr2 += i
                #countr2 = float(countr2)
                ok, frame = cap.read()
                if(ok == False) :
                    break
                cv2.imwrite('frame.jpg', frame)
                args.content = 'frame.jpg'
                #args.interpolation_weights = [countr, countr2]
                evaluate_network(args)
                stylised_frame = cv2.imread('stylized_image.jpg')
                stylised_frame = cv2.resize(stylised_frame, (frame_width, frame_height), interpolation = cv2.INTER_AREA) 
                out.write(stylised_frame)
            cap.release()
            out.release()
            cv2.destroyAllWindows()
