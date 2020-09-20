import os
from pathlib import Path


def kill_runtime():
    import os
    os.kill(os.getpid(), 9)


def download_youtube_video(youtube_id, youtube_format, youtube_file):
    try:
        import youtube_dl
    except:
        print('installing youtube_dl')
        os.system('pip install youtube-dl')
        import youtube_dl

    if not Path(youtube_file).exists():
        youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'
        print(f'downloading {youtube_url}')
        os.system(f'youtube-dl {youtube_url} -f {youtube_format} -o video.mp4')
    else:
        print(f'youtube video already present')


def download_google_drive_file(file_id, file_path):
    import gdown
    if not Path(file_path).exists():
        print('downloading model weights')
        gdown.download(f'https://drive.google.com/uc?id={file_id}', file_path, quiet=False)


def _pip(args):
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip"] + args)
    # , stdout=sys.stdout, stderr=sys.stderr) <-- doesn't work with Colab


def install_detectron2_colab():
    try:
        import detectron2
        return
    except:
        print('installing detectron2')

    commands = """
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
!pip install cython pyyaml==5.1
!pip install -U git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

!pip uninstall -y opencv-python
!pip uninstall -y opencv-contrib-python
!pip install opencv-python==3.4.2.16
!pip install opencv-contrib-python==3.4.2.16
!pip install opencv-contrib-python-nonfree

!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
    """
    for c in (c for c in commands.split('\n') if len(c) > 0):
        header = '!pip '
        if c.startswith(header):
            args = c[len(header):].split(' ')
            _pip(args)


def install_gmqtt():
    try:
        import gmqtt
        return
    except:
        print('installing gmqtt')
    _pip(['install', 'gmqtt'])


def video_frames(filename, start_frame=0, count=None, apply_COLOR_RGB2BGR=True, frame_list=None):
    import cv2
    import itertools

    video = cv2.VideoCapture(filename)
    if start_frame > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if frame_list and count is None:
        count = len(frame_list)

    for i in itertools.count():
        if i == count:
            break
        if frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_list[i])
        success, frame = video.read()
        if not success:
            if count is None:
                break
            else:
                raise Exception('Frame index out of bounds ' + (i + start_frame))
        if apply_COLOR_RGB2BGR:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        yield frame
    video.release()


frame_index_list = [2, 177, 764, 1676, 2831, 3212, 3874, 4165, 4357, 4677
    , 4823, 5247, 5967, 5978, 6574, 6688, 7029, 7527, 8018, 8319, 9017]


def new_detectron2_predictor(model_weights_file, NUM_CLASSES=2):
    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()

    import torch, torchvision

    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ('synth_train',)
    cfg.DATASETS.TEST = ('synth_val',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.WEIGHTS = model_weights_file
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return DefaultPredictor(cfg)


class VideoFrames:
    def __init__(self, video_input):
        import cv2
        self.video = cv2.VideoCapture(video_input)
        video = self.video
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_size = (self.width, self.height)

    def generator(self, start_frame=0, count=None, apply_COLOR_RGB2BGR=True, frame_list=None,
                  enumerate_frame_index=False):
        import cv2
        import itertools
        if start_frame > 0:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if frame_list and count is None:
            count = len(frame_list)

        for i in itertools.count():
            if i == count:
                break
            if frame_list:
                frame_index = frame_list[i]
                self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            else:
                frame_index = start_frame + i
            success, frame = self.video.read()
            if not success:
                if count is None:
                    break
                else:
                    raise Exception('Frame index out of bounds ' + (i + start_frame))
            if apply_COLOR_RGB2BGR:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if enumerate_frame_index:
                yield frame_index, frame
            else:
                yield frame


class VideoWriter:
    def __init__(self, output_fname, frame_size=(1024, 768), fps=1):
        from pathlib import Path
        import cv2
        of = Path(output_fname)
        if of.exists():
            of.unlink()

        if output_fname.endswith('.avi'):
            encoder = 'XVID'
        elif output_fname.endswith('.mkv'):
            encoder = 'x264'
        elif output_fname.endswith('.mpg') or output_fname.endswith('.mpeg'):
            encoder = 'MPEG'
        else:
            encoder = 'XVID'

        self.output_file = cv2.VideoWriter(
            filename=output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG o XVID o x264)
            fourcc=cv2.VideoWriter_fourcc(*encoder),
            fps=fps,
            frameSize=frame_size,
            isColor=True,
        )

    def write(self, frame):
        self.output_file.write(frame)

    def release(self):
        self.output_file.release()
        self.output_file = None


if __name__ == '__main__':
    youtube_file = 'video.mp4'
    download_youtube_video('JZdqjtWsL0U', '"best[height=360]"', youtube_file)
