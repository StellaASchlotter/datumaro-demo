from datumaro import Dataset
import datumaro

def merge():
    # TODO automatically collect them in a list
    my_dataset1 = Dataset.import_from("coco_datasets/task_outdoor-stand-bottom-2023_06_29_16_20_30", "coco")
    my_dataset2 = Dataset.import_from("coco_datasets/task_outdoor-stand-top-2023_06_29_16_15_08", "coco")
    my_dataset3 = Dataset.import_from("coco_datasets/task_outdoor5-bottom-2023_06_29_17_10_01", "coco")
    my_dataset4 = Dataset.import_from("coco_datasets/task_outdoor5-top-2023_06_29_16_34_30", "coco")
    my_dataset5 = Dataset.import_from("coco_datasets/task_outdoor6-bottom-2023_06_29_08_51_47", "coco")
    my_dataset6 = Dataset.import_from("coco_datasets/task_outdoor6-top-2023_06_29_08_30_31", "coco")

    # TODO: unpack a list here
    merged = datumaro.HLOps.merge(my_dataset1, my_dataset2, my_dataset3, my_dataset4, my_dataset5, my_dataset6)

    splited_dataset = merged.transform(
        "random_split", splits=[("train", 0.9), ("val", 0.1)]
    )
    splited_dataset.export("test-ultra", "yolo_ultralytics", save_media=True)

merge()