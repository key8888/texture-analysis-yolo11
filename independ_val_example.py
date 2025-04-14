# 独立した検証用画像構成を作る


from pathlib import Path
import yaml

        
def make_independent_val_img_structure(output_dir_path: Path, imgs: Path, txts: Path, classes_txt: Path, notes_json: Path) -> None:
    file_names = []
    classes_txt_file = classes_txt.read_bytes()
    notes_json_file = notes_json.read_bytes()

    for item in imgs.iterdir():
        file_names.append(item.stem)
        
    for name in file_names:
        # create directory structure and write files
        folder_train_img = output_dir_path / f"no_{name}" / "train" / "images"
        folder_train_img.mkdir(parents=True, exist_ok=True)
        
        folder_train_lab = output_dir_path / f"no_{name}" / "train" / "labels"
        folder_train_lab.mkdir(parents=True, exist_ok=True)
        
        folder_val_img = output_dir_path / f"no_{name}" / "val" / "images"
        folder_val_img.mkdir(parents=True, exist_ok=True)
        
        folder_val_lab = output_dir_path / f"no_{name}" / "val" / "labels"
        folder_val_lab.mkdir(parents=True, exist_ok=True)
        
        (folder_train_img.parent / classes_txt.name).write_bytes(classes_txt_file)
        (folder_train_img.parent / notes_json.name).write_bytes(notes_json_file)
        (folder_val_img.parent / classes_txt.name).write_bytes(classes_txt_file)
        (folder_val_img.parent / notes_json.name).write_bytes(notes_json_file)

        # copy images and labels to train and val folders
        for item in imgs.iterdir():
            if item.stem != name:
                data = item.read_bytes()
                (folder_train_img / item.name).write_bytes(data)
            else:
                data = item.read_bytes()
                (folder_val_img / item.name).write_bytes(data)
                
        for item in txts.iterdir():
            if item.stem != name:
                data = item.read_bytes()
                (folder_train_lab / item.name).write_bytes(data)
            else:
                data = item.read_bytes()
                (folder_val_lab / item.name).write_bytes(data)
                
        # create YAML file for ever train and val folder
        train = folder_train_img.parent.resolve()
        val = folder_train_img.parent.resolve()
        
        data = {
            'train' : str(train),
            'val' : str(val),
            'nc' : 1,
            'names' : ['y2o3']
        }
        with open(output_dir_path / f"no_{name}" / "custom_dataset.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True)
            

dataset_source = Path('YOLO_dataset_zip', 'project-6-at-2025-03-23-20-14-00444e1f')
imgs = dataset_source / 'images'
txts = dataset_source / 'labels'
classes_txt = dataset_source / 'classes.txt'
notes_json = dataset_source / 'notes.json'
output_dir_path = Path('fine')

make_independent_val_img_structure(output_dir_path, imgs, txts, classes_txt, notes_json)