import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
import os
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def image_process(image_path, result_path, class_name, w, h, e):
    class_path = os.path.join(image_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(result_path, class_name)
    if not os.path.exists(dst_class_path):
        os.makedirs(dst_class_path)

    for folder_name in os.listdir(class_path):
        dst_directory_path = os.path.join(dst_class_path, folder_name)

        folder_path = os.path.join(class_path,folder_name)
        if os.path.isfile(folder_path):
            image = common.read_imgfile(folder_path, None, None)
            if image is None:
                logger.error('Image can not be read, path=%s' % folder_name)
                sys.exit(-1)

            t = time.time()
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            elapsed = time.time() - t

            logger.info('inference image: %s in %.4f seconds.' % (folder_name, elapsed))

            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(dst_directory_path, image)
        else:
            if not os.path.exists(dst_directory_path):
                os.makedirs(dst_directory_path)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path,file_name)
                image = common.read_imgfile(file_path, None, None)
                if image is None:
                    logger.error('Image can not be read, path=%s' % file_name)
                    sys.exit(-1)

                t = time.time()
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
                elapsed = time.time() - t

                logger.info('inference image: %s in %.4f seconds.' % (file_name, elapsed))

                image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(dst_directory_path,file_name), image)
            # plt.figure()
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.savefig(os.path.join(dst_directory_path,file_name),bbox_inches = 'tight',pad_inches = 0)
            # bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            # bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

            # # show network output
            # a = fig.add_subplot(2, 2, 2)
            # plt.imshow(bgimg, alpha=0.5)
            # tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
            # plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
            # plt.colorbar()

            # tmp2 = e.pafMat.transpose((2, 0, 1))
            # tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            # tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

            # a = fig.add_subplot(2, 2, 3)
            # a.set_title('Vectormap-x')
            # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            # plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
            # plt.colorbar()

            # a = fig.add_subplot(2, 2, 4)
            # a.set_title('Vectormap-y')
            # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            # plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
            # plt.colorbar()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image_path', type=str, default='./images/')
    parser.add_argument('--result_path', type=str, default='./images_pose/')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !

    image_path = args.image_path
    result_path = args.result_path

    for class_name in os.listdir(image_path):
        image_process(image_path, result_path, class_name, w, h, e)
    

