def cv2_imshow(img, title=None, figsize=(15, 8)):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    if title:
        plt.gcf().suptitle(title, fontsize=20)
    plt.imshow(img)
    plt.show()


import skimage.io
from detectron2.utils.visualizer import Visualizer

import numpy as np
import cv2

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import sys
from skimage import metrics
from PIL import Image, ImageDraw, ImageFont


# Library from: https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/

def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)

    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0, 256])
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0, 256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0, 256])

    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)

    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)

    # Put the image back together
    image_after_matching = cv2.merge([
        blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching


def checkInstance(instance, template, detector, matcher, kp_template, des_template, lowe_thresh=0.9):
    kp_instance, des_instance = detector.detectAndCompute(instance, None)
    ###cv2.drawKeypoints(instance, kp_instance, None)

    # Matches between template and target
    matches = matcher.knnMatch(queryDescriptors=des_instance, trainDescriptors=des_template, k=2)  # Best 2 matches
    # print("Instance to template matches:", len(matches))

    # Lowe ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < lowe_thresh * n.distance:
            good_matches.append(m)
    # print("total matches", len(matches), 'good matches', len(good_matches))

    if len(good_matches) < 8:  ## FINE TUNE!!!
        # print("Too less good matches")
        return False, None

    """if len(good_matches) / len(matches) < 0.2:  ## FINE TUNE!!!
      return False"""

    instance_pts = np.array([kp_instance[p.queryIdx].pt for p in good_matches], dtype=float).reshape(-1, 1, 2)
    template_pts = np.array([kp_template[p.trainIdx].pt for p in good_matches], dtype=float).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(instance_pts, template_pts, cv2.RANSAC, ransacReprojThreshold=3.0)
    try:
        Hinv = np.linalg.inv(H)
    except Exception as e:
        # raise Exception(f'Original exception type:{type(e)} ' + str(e))
        return False, None

    inlier_mask = inlier_mask.flatten().astype(bool)
    n_inlier = np.count_nonzero(inlier_mask)
    matchesMask = inlier_mask.flatten().tolist()

    h, w, _ = template.shape
    bb = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype=float).reshape(-1, 1, 2)  # Bounding box
    Hbb = cv2.perspectiveTransform(bb, Hinv)  # Apply homography to bounding box

    inlier_pts = instance_pts[inlier_mask]

    sift_bb = [Hbb[i, 0].tolist() for i in range(4)]

    polygon = Polygon(sift_bb)

    # Check if matches are inside the transformed bounding box
    outsiders = 0  # Number of points outside the transformed bounding box
    for p in inlier_pts:
        point = Point(p[0, 0], p[0, 1])
        if not polygon.contains(point):
            outsiders += 1

    out_proportion = 0.5  ### SET AS METHOD PARAMETERS !!! 0.5
    ssim_thresh = 0.4  ### SET AS METHOD PARAMETERS !!!

    # With too less correspondences the template cannot be considered matchable
    # print('outsiders', outsiders, 'n_inlier', n_inlier, 'proportion', outsiders / n_inlier)
    if (outsiders / n_inlier < out_proportion):
        rectified = cv2.warpPerspective(instance, H, (template.shape[1], template.shape[0]))

        restored = match_histograms(rectified, template)

        ssim = metrics.structural_similarity(template, restored, multichannel=True)

        if ssim >= ssim_thresh:
            # print(ssim)

            """fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(rectified)
            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(restored)
            ax = fig.add_subplot(1, 3, 3)
            ax.imshow(template)"""

            # print('accepted')
            return True, sift_bb
        else:
            # print('not accepted, because of structural similarity')
            return False, sift_bb
    else:
        # print('not accepted because of outliers / n_inliers proportion')
        return False, None


import cv2

detector = cv2.xfeatures2d.SIFT_create()  # SIFT detector object
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)  # FLANN matcher object


def fittAbbestia(target, instances, templates):
    import numpy as np
    """
    target: target image
    instances: detectron2 outputs["instances"].to("cpu")
    templates: dictionary of templates
    """
    target = np.array(target)[:, :, :3]

    accepted = []
    boxes = []  # output bounding boxes
    sift_boxes = []
    box_labels = []  # output labels for bounding boxes

    labels = list(templates.keys())  # all labels available

    # Templates sift features
    template_feats = {}
    for label in labels:
        template = np.array(templates[label])[:, :, :3]
        kp_template, des_template = detector.detectAndCompute(template, None)
        ###cv2.drawKeypoints(template, kp_template, None)
        template_feats[label] = (kp_template, des_template)

    #
    def enlarge_box(box, increase=0.25):
        max_height, max_width = target.shape[:2]
        h = int((box[3] - box[1]) * increase)
        w = int((box[2] - box[0]) * increase)
        r0 = np.clip(box[1] - h, 0, max_height)
        r1 = np.clip(box[3] + h, 0, max_height)
        c0 = np.clip(box[0] - w, 0, max_width)
        c1 = np.clip(box[2] + h, 0, max_width)
        return (c0, r0, c1, r1)

    for i in range(0, len(instances)):
        box = instances[i]['pred_boxes']

        box = enlarge_box(box)
        patch = target[box[1]:box[3], box[0]:box[2], :]

        # ###print('box',box,'target.shape',target.shape,'instance.shape',instance.shape)
        # target.shape (720, 1280, 3) box [460  34 618 119] instance.shape (85, 158, 3)
        def is_bb_inside_patch(bb):
            all_x, all_y = zip(*bb)
            x_bad = [x for x in all_x if x < 0 or x > patch.shape[1] * 1.1]
            y_bad = [y for y in all_y if y < 0 or y > patch.shape[0] * 1.1]
            ok = len(x_bad) == 0 and len(y_bad) == 0
            # if not ok:
            #     print('sift bb not ok', patch.shape, x_bad, y_bad, bb)
            return ok

        def transform_sift_point(p):
            x = int(p[0])
            y = int(p[1])
            return (x + box[0], y + box[1])

        for label in labels:
            template = np.array(templates[label])[:, :, :3]
            kp_template, des_template = template_feats[label]
            ok, sift_bb = checkInstance(patch, template, detector, matcher, kp_template, des_template)
            if ok:
                ok = is_bb_inside_patch(sift_bb)
            sift_bb = [transform_sift_point(p) for p in sift_bb] if ok else None
            sift_boxes.append(sift_bb)

            # if ok: # we always give a result
            accepted.append(ok)
            boxes.append(box)
            box_labels.append(label)

    return accepted, boxes, sift_boxes, box_labels


def putText(image, label, point, color=(255, 255, 255)):
    bottomLeftCornerOfText = point
    fontScale = 1
    lineType = 2

    cv2.putText(image, label,
                bottomLeftCornerOfText,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                lineType)


def draw_bbs(target, accepted, boxes, labels):
    target = out_img = Image.fromarray(target)
    draw = ImageDraw.Draw(target)

    for ok, label, box in zip(accepted, labels, boxes):
        # if not ok:
        # continue
        draw.line((box[0], box[1]) + (box[0], box[3]), fill=(0, 255, 0), width=6)
        draw.line((box[0], box[1]) + (box[2], box[1]), fill=(0, 255, 0), width=6)
        draw.line((box[2], box[3]) + (box[2], box[1]), fill=(0, 255, 0), width=6)
        draw.line((box[2], box[3]) + (box[0], box[3]), fill=(0, 255, 0), width=6)
        # draw.text((box[0] + 5, box[1] + 5), label, fill=(0, 255, 0)) # , font=font, anchor=None
    target = np.array(target)
    idx = -1
    for ok, label, box in zip(accepted, labels, boxes):
        idx += 1
        okstr = 'ok' if ok else 'ko'
        putText(target, f'{idx} {label} {okstr}', (box[0] + 5, box[1] + 25), (0, 255, 0))

    return target


def drawSiftBoxes(target, accepted, boxes, labels, color=(255, 255, 255), write_index=False):
    target = Image.fromarray(target)
    draw = ImageDraw.Draw(target)
    idx = -1
    for ok, label, box in zip(accepted, labels, boxes):
        idx += 1
        if not ok:
            continue
        # print('sift foundings box', idx, 'coords', box)
        for i in range(len(box)):
            a = box[i]
            b = box[(i + 1) % len(box)]
            draw.line((a[0], a[1], b[0], b[1]), fill=color, width=6)

    idx = -1
    target = np.array(target)
    for ok, label, box in zip(accepted, labels, boxes):
        idx += 1
        if not ok:
            continue
        a = box[0]
        idxstr = str(idx) if write_index else ''
        putText(target, f'{idxstr}{label}', (a[0], a[1] - 9), color)
    return target


def drawDetectronOutput(target, instances, colors=[(0, 0, 255)], write_index=False):
    target = Image.fromarray(target)
    draw = ImageDraw.Draw(target)
    boxes = [instances[i]['pred_boxes'] for i in range(len(instances))]
    classes = [instances[i]['pred_classes'] for i in range(len(instances))]

    def get_color(index):
        class_index = classes[index]
        return colors[class_index % len(colors)]

    for i, box in enumerate(boxes):
        color = get_color(i)
        draw.line((box[0], box[1]) + (box[0], box[3]), fill=color, width=6)
        draw.line((box[0], box[1]) + (box[2], box[1]), fill=color, width=6)
        draw.line((box[2], box[3]) + (box[2], box[1]), fill=color, width=6)
        draw.line((box[2], box[3]) + (box[0], box[3]), fill=color, width=6)
    target = np.array(target)
    if write_index:
        for i, box in enumerate(boxes):
            idxstr = str(i) if write_index else ''
            putText(target, f'{idxstr}{classes[i]}', (box[0] + 5, box[1] + 25), get_color(i))
    return target


def clean_predictor_output(outputs):
    def extract(i):
        return {
            'pred_boxes': list(i.get('pred_boxes').tensor.squeeze().numpy().astype(int)),
            'pred_classes': i.get('pred_classes').numpy()[0]
        }

    instances = outputs["instances"].to("cpu")
    instances.remove('pred_masks')

    res = [extract(instances[i]) for i in range(len(instances))]
    return res
