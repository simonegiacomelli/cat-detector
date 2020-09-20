import s01.colab_utils as u
import cv2

youtube_file = 'video.avi'
u.download_youtube_video('JZdqjtWsL0U', '"best[height=360]"', youtube_file)

video = cv2.VideoCapture(youtube_file)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

output_file = cv2.VideoWriter(
    filename='video-output.avi',
    # some installation of opencv may not support x264 (due to its license),
    # you can try other format (e.g. MPEG o XVID o x264)
    fourcc=cv2.VideoWriter_fourcc(*"XVID"),
    fps=frames_per_second,
    frameSize=(width, height),
    isColor=True,
)

detectron2 = 'Detectron2 bounding boxes'
sift_ransac = 'After SIFT and RANSAC postprocessing'


def put_text(lbl, bottomLeftCornerOfText, fontColor=(255, 255, 255)):
    fontScale = 1

    lineType = 2
    cv2.putText(frame, lbl,
                bottomLeftCornerOfText,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                fontColor,
                lineType)


vert_displ = 30
color = (200, 200, 200)
for idx, frame in enumerate(u.video_frames(youtube_file, 0, None, apply_COLOR_RGB2BGR=False)):
    # put_text(f'{idx}'.rjust(4, '0'), (width - 90, height - 10))
    put_text(detectron2, (5, vert_displ), fontColor=color)
    put_text(sift_ransac, (5, height // 2 + vert_displ), fontColor=color)
    output_file.write(frame)

output_file.release()
