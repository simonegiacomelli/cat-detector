from s01.coco_image_viewer_lib import CocoDataset


def auto_complete(file_path):
    return file_path


def main():
    dataset_folder = auto_complete('../datasets/f4/synth_dataset_training/output')
    instances_json_path = "%s/coco_annotations.json" % dataset_folder
    images_path = "%s/images" % dataset_folder
    coco_dataset = CocoDataset(instances_json_path, images_path)
    coco_dataset.display_info()
    coco_dataset.display_licenses()
    coco_dataset.display_categories()
    # this number comes from the json
    # beware! that not all images are present in the folder
    # look into folder {images_path}
    html = coco_dataset.display_image(next(iter(coco_dataset.images)))
    output_html = 'temporary_output.html'
    with open(output_html, 'w') as f:
        f.write(html)
        # IPython.display.HTML(html)
    import os
    import webbrowser
    webbrowser.open('file://' + os.path.realpath(output_html))


if __name__ == '__main__':
    main()
