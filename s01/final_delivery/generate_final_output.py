#! /usr/bin/env python3
import os

os.system('cd fantastic4v2 && git pull || git clone https://github.com/simonegiacomelli/fantastic4v2')
import fantastic4v2

from s01.colab_utils import *

# install_detectron2_colab()

youtube_file = 'video.mp4'
download_youtube_video('JZdqjtWsL0U', '"best[height=720]"', youtube_file)

model_weights_file = 'model_final.pth'
# download_google_drive_file('17hCPzRdaN8PwTN1P6MLEvur9xGxzXBLw',model_weights_file)
download_google_drive_file('1oZ4im--pS_o7y2ly5lXccgjT40A7kPts', model_weights_file)

predictor = new_detectron2_predictor(model_weights_file)

from s01.detect_utils import *

import skimage.io
from detectron2.utils.visualizer import Visualizer

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import sys
from skimage import metrics
from PIL import Image, ImageDraw, ImageFont
import tqdm

datasets = './fantastic4v2/datasets'
target = skimage.io.imread('%s/no_labels/target2.png' % datasets)[:, :, :3]
# emirates = skimage.io.imread('./fantastic4v2/datasets/f4/synth_dataset_training/input/foregrounds/emirates/emirates1/Tempate2-a.png')[:,:,:3]
foregrounds = '/f4/synth_dataset_training/input/foregrounds'
emirates = skimage.io.imread('%s%s/emirates/emirates1/Emirates_logo_red_background2.png' %
                             (datasets, foregrounds))[:, :, :3]
arsenal1 = skimage.io.imread('%s%s/arsenal/arsenal1/arsenal1.png' % (datasets, foregrounds))[:, :, :3]
arsenal2 = skimage.io.imread('%s%s/arsenal/arsenal1/arsenal2.png' % (datasets, foregrounds))[:, :, :3]

# templates = {"emirates": emirates,'arsenal':arsenal} # Dict of templates you want to detect
# templates = {"emirates": emirates} # Dict of templates you want to detect
# templates = {"arsenal": arsenal1}
all_templates = [{"arsenal": arsenal1}, {"emirates": emirates}]
all_colors = [(0, 255, 0), (0, 0, 255)]
# version 2.0
frame_index_list = [2, 177, 764, 1676, 2831, 3212, 3874, 4165, 4357, 4677, 4823, 5247, 5967, 5978, 6574, 6688, 7029,
                    7527, 8018, 8319, 9017][:6]
# print('frame indexes chosen', frame_index_list)
video_frames = VideoFrames(youtube_file)
frame_size = (video_frames.width, video_frames.height * 2)
video_writer = VideoWriter('video-output.avi', frame_size=frame_size, fps=video_frames.fps)
# video_frames_gen = video_frames(youtube_file,frame_list=frame_index_list)

video_frames_gen = video_frames.generator(start_frame=0, enumerate_frame_index=True)
for frame_index, frame in tqdm.tqdm(video_frames_gen, total=video_frames.frame_count):

    instances = clean_predictor_output(predictor(frame))
    fitt_results = []
    for template_idx, templates in enumerate(all_templates):
        fitt_results.append((template_idx,) + fittAbbestia(frame, instances, templates))

        # draw_bbs(target, accepted, sift_search_boxes, labels), 'sift bounding box'

    sift_findings_frame = frame
    for template_idx, accepted, sift_search_boxes, sift_boxes, labels in fitt_results:
        color = all_colors[template_idx]
        sift_findings_frame = drawSiftBoxes(sift_findings_frame, accepted, sift_boxes, labels, color=color)

    detectron_frame = drawDetectronOutput(frame, instances, colors=all_colors)

    frame_to_write = np.concatenate((detectron_frame, sift_findings_frame), axis=0)

    video_writer.write(frame=cv2.cvtColor(frame_to_write, cv2.COLOR_RGB2BGR))

video_writer.release()
