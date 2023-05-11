from datumaro import Dataset

my_dataset = Dataset.import_from("my_coco_dataset", "coco")

# TODO how can we create the ultralytics format and use 100% for training?
splited_dataset = my_dataset.transform(
    "random_split", splits=[("train", 0.5), ("val", 0.5)]
)
splited_dataset.export("my_ultralytics_dataset", "yolo_ultralytics", save_media=True)