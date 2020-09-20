#!/usr/bin/env python3
import json
import math
import random
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


def apply_random_homography(img_pil, max_scaling_factor=1.0):
    img_np = np.array(img_pil)

    h, w = img_np.shape[:2]

    edges = [np.array([0, 0, 1]), np.array([w, 0, 1]), np.array([w, h, 1]), np.array([0, h, 1])]

    theta = random.uniform(0, np.pi)  # Rotation angle
    phi = random.uniform(0, np.pi)
    # theta = np.pi / 2
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    cosPhi = np.cos(phi)
    sinPhi = np.sin(phi)

    sx = random.uniform(0.25, max_scaling_factor)  # Scaling x
    sy = random.uniform(0.25, max_scaling_factor)  # Scaling y
    p1 = random.uniform(0., .001)
    p2 = random.uniform(0., .001)

    R_theta = np.array([[cosTheta, -sinTheta],
                        [sinTheta, cosTheta]])  # Rotation theta matrix

    D = np.array([[sx, 0],
                  [0, sy]])  # Non-isotropic scaling

    R_phi = np.array([[cosPhi, -sinPhi],
                      [sinPhi, cosPhi]])  # Rotation phi matrix

    R_minus_phi = np.array([[cosPhi, sinPhi],
                            [-sinPhi, cosPhi]])  # Rotation minus phi matrix

    A = R_theta @ R_minus_phi @ D @ R_phi  # Affinity (rotation + non-isotropic scaling)

    Hp = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [p1, p2, 0]])  # Perspective transformation

    H = np.zeros((3, 3))
    H[:2, :2] = A
    H[2, 2] = 1.

    H = H + Hp

    # Compute translation matrix
    new_edges = [H @ e for e in edges]
    new_edges_norm = np.asarray([(e / e[2])[:2] for e in new_edges])  ### CAST edges to int!!!
    tx = new_edges_norm[:, 0].min()
    ty = new_edges_norm[:, 1].min()

    b_w = math.ceil(new_edges_norm[:, 0].max() - tx)
    b_h = math.ceil(new_edges_norm[:, 1].max() - ty)

    T = np.array([[1, 0, -tx],
                  [0, 1, -ty],
                  [0, 0, 1]])  # Translation matrix

    # Inlcude translation in H
    H = T @ H

    # Recompute edges (final)
    new_edges = [H @ e for e in edges]
    new_edges_norm = np.asarray([(e / e[2])[:2] for e in new_edges])

    out_np = cv2.warpPerspective(img_np, H, (b_w, b_h))
    out_pil = Image.fromarray(out_np)

    return out_pil, new_edges_norm


class AnnotationsJsonUtils:
    """ Creates a JSON definition file for coco dataset
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.images = []
        self.annotations = []
        self.categories = {}

    def get_category_id(self, category, super_category):
        s = str(category) + '-' + str(super_category)
        if s not in self.categories:
            self.categories[s] = {
                'id': len(self.categories) + 1,
                'name': category,
                'supercategory': super_category
            }
        return self.categories[s]['id']

    def get_image_id(self, file_name, height, width):
        image_id = len(self.images) + 1
        self.images.append({
            'id': image_id,
            'file_name': file_name,
            'height': height,
            'width': width
        })
        return image_id

    def get_annotation_id(self, category, super_category, image_id, bbox, segmentation):
        category_id = self.get_category_id(category, super_category)
        annotation_id = len(self.annotations) + 1
        self.annotations.append({
            'id': annotation_id,
            'segmentation': segmentation,  # list of lists
            'iscrowd': 0,
            # 'area': 1234.5,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox
        })
        return annotation_id

    def save(self):
        obj = {
            'images': self.images,
            'categories': [item for key, item in self.categories.items()],
            'annotations': self.annotations
        }

        # Write the JSON output file
        file = (Path(self.output_dir) / 'coco_annotations.json').open('w')
        json.dump(obj, file, indent=4)
        file.close()


# Credits for original code of this class: https://github.com/akTwelve/cocosynth
class ImageComposition():
    """ Composes images together in random ways, applying transformations to the foreground to create a synthetic
        combined image.
    """

    def __init__(self, args):
        self.args = args
        self.allowed_output_types = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types = ['.png', '.jpg', '.jpeg']
        self.zero_padding = 8  # 00000027.png, supports up to 100 million images

    def _validate_and_process_args(self):

        # Validate the count
        assert self.args.count > 0, 'count must be greater than 0'
        self.count = self.args.count

        # Validate the width and height
        assert self.args.width >= 64, 'width must be greater than 64'
        self.width = self.args.width
        assert self.args.height >= 64, 'height must be greater than 64'
        self.height = self.args.height

        # Validate and process the output type
        if self.args.output_type is None:
            self.output_type = '.jpg'  # default
        else:
            if self.args.output_type[0] != '.':
                self.output_type = f'.{self.args.output_type}'
            assert self.output_type in self.allowed_output_types, f'output_type is not supported: {self.output_type}'

        # Validate and process output and input directories
        self._validate_and_process_output_directory()
        self._validate_and_process_input_directory()

    def _validate_and_process_output_directory(self):
        self.output_dir = Path(self.args.output_dir)
        self.images_output_dir = self.output_dir / 'images'

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_output_dir.mkdir(exist_ok=True)

    def _validate_and_process_input_directory(self):
        self.backgrounds_dir = Path(self.args.backgrounds_dir)
        self.foregrounds_dir = Path(self.args.foregrounds_dir)
        assert self.backgrounds_dir.exists(), f'backgrounds_dir does not exist: {self.backgrounds_dir}'

        self._validate_and_process_foregrounds()
        self._validate_and_process_backgrounds()

    def _validate_and_process_foregrounds(self):

        self.foregrounds_dict = dict()

        for super_category_dir in self.foregrounds_dir.iterdir():
            if not super_category_dir.is_dir():
                raise Exception(f'file found in foregrounds directory '
                                f'(expected super-category directories), '
                                f'ignoring: {super_category_dir}')
                continue

            # This is a super category directory
            for category_dir in super_category_dir.iterdir():
                if not category_dir.is_dir():
                    raise Exception(
                        f'file found in super category directory (expected category directories), ignoring: {category_dir}')
                    continue

                # This is a category directory
                for image_file in category_dir.iterdir():
                    if not image_file.is_file():
                        raise Exception(
                            f'a directory was found inside a category directory, ignoring: {str(image_file)}')
                        continue
                    if image_file.suffix != '.png':
                        raise Exception(f'foreground must be a .png file, skipping: {str(image_file)}')
                        continue

                    # Valid foreground image, add to foregrounds_dict
                    super_category = super_category_dir.name
                    category = category_dir.name

                    if super_category not in self.foregrounds_dict:
                        self.foregrounds_dict[super_category] = dict()

                    if category not in self.foregrounds_dict[super_category]:
                        self.foregrounds_dict[super_category][category] = []

                    self.foregrounds_dict[super_category][category].append(image_file)

        assert len(self.foregrounds_dict) > 0, 'no valid foregrounds were found'

    def _validate_and_process_backgrounds(self):
        self.backgrounds = []
        for image_file in self.backgrounds_dir.iterdir():
            if not image_file.is_file():
                warnings.warn(f'a directory was found inside the backgrounds directory, ignoring: {image_file}')
                continue

            if image_file.suffix not in self.allowed_background_types:
                warnings.warn(
                    f'background must match an accepted type {str(self.allowed_background_types)}, ignoring: {image_file}')
                continue

            # Valid file, add to backgrounds list
            self.backgrounds.append(image_file)

        assert len(self.backgrounds) > 0, 'no valid backgrounds were found'

    def _generate_images(self):

        print(f'Generating {self.args.name} dataset with COCO annotations...')
        shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=False)

        anns = AnnotationsJsonUtils(self.output_dir)

        # Create all images (with tqdm to have a progress bar)
        for i in tqdm(range(self.count)):
            # Randomly choose a background
            background_path = random.choice(self.backgrounds)

            num_foregrounds = random.randint(1, self.args.max_foregrounds)
            foregrounds = []
            self.preload_categories(anns)
            for fg_i in range(num_foregrounds):
                # Randomly choose a foreground
                super_category = random.choice(list(self.foregrounds_dict.keys()))
                category = random.choice(list(self.foregrounds_dict[super_category].keys()))
                foreground_path = random.choice(self.foregrounds_dict[super_category][category])

                foregrounds.append({
                    'super_category': super_category,
                    'category': category,
                    'foreground_path': foreground_path
                })

            # Compose foregrounds and background
            composite, mask = self._compose_images(foregrounds, background_path)

            # Create the file name (used for both composite and mask)
            save_filename = f'{i:0{self.zero_padding}}'  # e.g. 00000023.jpg

            # Save composite image to the images sub-directory
            composite_filename = f'{save_filename}{self.output_type}'  # e.g. 00000023.jpg
            composite_path = self.output_dir / 'images' / composite_filename  # e.g. my_output_dir/images/00000023.jpg
            composite = composite.convert('RGB')  # remove alpha
            composite.save(composite_path)

            image_id = anns.get_image_id(composite_filename, composite.size[0], composite.size[1])
            for fg in foregrounds:
                anns.get_annotation_id(fg['category'], fg['super_category'], image_id, fg['bbox'], fg['segmentation'])

        anns.save()

    def _compose_images(self, foregrounds, background_path):
        background = Image.open(background_path)
        background = background.convert('RGBA')

        # Crop background to desired size (self.width x self.height), randomly positioned
        bg_width, bg_height = background.size
        max_crop_x_pos = bg_width - self.width
        max_crop_y_pos = bg_height - self.height
        assert max_crop_x_pos >= 0, f'desired width, {self.width}, is greater than background width, {bg_width}, for {str(background_path)}'
        assert max_crop_y_pos >= 0, f'desired height, {self.height}, is greater than background height, {bg_height}, for {str(background_path)}'
        crop_x_pos = random.randint(0, max_crop_x_pos)
        crop_y_pos = random.randint(0, max_crop_y_pos)
        composite = background.crop((crop_x_pos, crop_y_pos, crop_x_pos + self.width, crop_y_pos + self.height))
        composite_mask = Image.new('RGB', composite.size, 0)

        for fg in foregrounds:
            fg_path = fg['foreground_path']

            # Perform transformations
            fg_image, edges_rot = self._transform_foreground(fg, fg_path)
            # Choose a random x,y position for the foreground
            max_x_position = composite.size[0] - fg_image.size[0]
            max_y_position = composite.size[1] - fg_image.size[1]
            assert max_x_position >= 0 and max_y_position >= 0, \
                f'foreground {fg_path} is too big ({fg_image.size[0]}x{fg_image.size[1]}) for the requested output size ({self.width}x{self.height}), check your input parameters'
            paste_position = (random.randint(0, max_x_position), random.randint(0, max_y_position))
            fg['bbox'] = list(paste_position + fg_image.size)
            px = paste_position[0]
            py = paste_position[1]
            fg['segmentation'] = [[item for point in [(p[0] + px, p[1] + py) for p in edges_rot] for item in point]]
            # Create a new foreground image as large as the composite and paste it on top
            new_fg_image = Image.new('RGBA', composite.size, color=(0, 0, 0, 0))
            new_fg_image.paste(fg_image, paste_position)

            # Extract the alpha channel from the foreground and paste it into a new image the size of the composite
            alpha_mask = fg_image.getchannel(3)
            new_alpha_mask = Image.new('L', composite.size, color=0)
            new_alpha_mask.paste(alpha_mask, paste_position)
            composite = Image.composite(new_fg_image, composite, new_alpha_mask)

            # Grab the alpha pixels above a specified threshold
            alpha_threshold = 200
            mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
            uint8_mask = np.uint8(mask_arr)  # This is composed of 1s and 0s

        return composite, composite_mask

    def _transform_foreground(self, fg, fg_path):

        fg_image = Image.open(fg_path)

        fg_image, edges_rot = apply_random_homography(fg_image)

        # Adjust foreground brightness
        brightness_factor = random.random() * .4 + .7  # Pick something between .7 and 1.1
        enhancer = ImageEnhance.Brightness(fg_image)
        fg_image = enhancer.enhance(brightness_factor)

        return fg_image, edges_rot

    def main(self):
        self._validate_and_process_args()
        self._generate_images()

    def preload_categories(self, anns):
        fg = self.foregrounds_dict
        classes = sorted([(sc, cl) for sc in fg for cl in fg[sc]])
        for cl in classes:
            anns.get_category_id(cl[1], cl[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic Dataset generation")
    # parser.add_argument("--input_dir", type=str, dest="input_dir", required=True, help="The input directory. \
    #                     This contains a 'backgrounds' directory of pngs or jpgs, and a 'foregrounds' directory which \
    #                     contains supercategory directories (e.g. 'animal', 'vehicle'), each of which contain category \
    #                     directories (e.g. 'horse', 'bear'). Each category directory contains png images of that item on a \
    #                     transparent background (e.g. a grizzly bear on a transparent background).")
    # parser.add_argument("--output_dir", type=str, dest="output_dir", required=True, help="The directory where images, masks, \
    #                     and json files will be placed")
    parser.add_argument("--count", type=int, dest="count", required=False, help="number of composed images to create",
                        default=10)
    # parser.add_argument("--width", type=int, dest="width", required=True, help="output image pixel width")
    # parser.add_argument("--height", type=int, dest="height", required=True, help="output image pixel height")
    # parser.add_argument("--output_type", type=str, dest="output_type", help="png or jpg (default)")

    args = parser.parse_args()

    training = {'name': 'training',
                'backgrounds_dir': '../datasets/f4/synth_dataset_training/input/backgrounds',
                'foregrounds_dir': '../datasets/f4/synth_dataset_training/input/foregrounds',
                'output_dir': '../datasets/f4/synth_dataset_training/output',
                'count': args.count,
                'width': 1080 / 2, 'height': 1080 / 2,
                'max_foregrounds': 1,
                'output_type': None}

    validation = {'name': 'validation',
                  'backgrounds_dir': '../datasets/f4/synth_dataset_validation/input/backgrounds',
                  'foregrounds_dir': '../datasets/f4/synth_dataset_validation/input/foregrounds',
                  'output_dir': '../datasets/f4/synth_dataset_validation/output',
                  'count': args.count,
                  'width': 1080 / 2, 'height': 1080 / 2,
                  'max_foregrounds': 1,
                  'output_type': None}


    class Arguments:
        pass


    all = [training, validation]
    for dictionary in all:
        d = Arguments()
        d.__dict__ = dictionary
        ImageComposition(d).main()
