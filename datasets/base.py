import os
import PIL
import json
import pickle
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from packaging import version
from transformers import CLIPTokenizer
import matplotlib.pyplot as plt

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


caption_templates = [
    "a photo of a {}",
    "a rendering of a {}",
    "the photo of a {}",
    "a photo of my {}",
    "a photo of the {}",
    "a photo of one {}",
    "a rendition of a {}",
]

class BaseDataset(Dataset):
    def __init__(self,
                 root=".",
                 repeats=1,
                 crop_size=512,
                 resize_size=512,
                 test_mode=False,
                 keep_class=None,
                 center_crop=False,
                 text_encoder=None,
                 load_class_path=None,
                 load_token_path=None,
                 load_pretrain_path=None,
                 interpolation="bicubic",
                 token_templates="<imagenet-{}>",
                 caption_templates=caption_templates,
                 **kwargs,
    ):
        self.root = root
        self.data_repeat = repeats if not test_mode else 1
        self.test_mode = test_mode
        self.keep_class = keep_class
        self.load_class_path = load_class_path
        self.load_token_path = load_token_path
        self.load_pretrain_path = load_pretrain_path
        self.token_templates = token_templates
        self.caption_templates = caption_templates

        self.train_pipelines = self.init_train_pipelines(
            center_crop=center_crop,
            resize_size=resize_size,
            crop_size=crop_size,
            interpolation=interpolation,
        )
        self.test_pipelines = self.init_test_pipelines(
            crop_size=crop_size,
            interpolation=interpolation,
        )

        print(f"INFO: {self.__class__.__name__}:\t load data.", flush=True)
        self.load_data()
        print(f"INFO: {self.__class__.__name__}:\t init samples.", flush=True)
        self.init_samples()
        print(f"INFO: {self.__class__.__name__}:\t init text encoders.", flush=True)
        self.init_text_encoder(text_encoder)

    def load_data(self):
        class_file = os.path.join(self.root, 'ILSVRC2012_list', 'LOC_synset_mapping.txt')
        self.categories = []
        with open(class_file, 'r') as f:
            discriptions = f.readlines()  # "n01882714 koala..."
            for id, line in enumerate(discriptions):
                tag, description = line.strip().split(' ', maxsplit=1)
                self.categories.append(description)
        self.num_classes = len(self.categories)

        self.names = []
        self.labels = []
        self.bboxes = []
        self.pred_logits = []
        self.image_paths = []

        data_file = os.path.join(self.root, 'ILSVRC2012_list', 'train.txt')
        image_dir = os.path.join(self.root, 'train')
        if self.test_mode:
            data_file = os.path.join(self.root, 'ILSVRC2012_list', 'val_folder_new.txt')
            image_dir = os.path.join(self.root, 'val')

        with open(data_file) as f:
            datamappings = f.readlines() # "n01440764/n01440764_10026.JEPG 0"
            for id, line in enumerate(datamappings):
                info = line.strip().split()
                self.names.append(info[0][:-5]) # "n01440764/n01440764_10026"
                self.labels.append(int(info[1])) # "0"
                if self.test_mode:
                    self.bboxes.append(np.array(list(map(float, info[2:]))).reshape(-1, 4))
        if self.keep_class is not None:
            self.filter_classes()
        self.pred_logits = None
        if self.test_mode:
            with open(self.load_class_path, 'r') as f:
                name2result = json.load(f)
                self.pred_logits = [torch.Tensor(name2result[name]['pred_scores']) for name in self.names]
        self.image_paths = [os.path.join(image_dir, name + '.JPEG') for name in self.names]
        self.num_images = len(self.labels)

    def init_samples(self):
        # format tokens by category
        def select_meta_tokens(cats=[], tokenizer=None):
            for c in cats:
                token_ids = tokenizer.encode(c, add_special_tokens=False)
                if len(token_ids) == 1:  # has exist token to indicate input token
                    return c, token_ids[-1], True
            token_ids = tokenizer.encode(cats[0], add_special_tokens=False)
            token = tokenizer.decode(token_ids[-1])  # only use the final one
            return token, token_ids[-1], False

        tokenizer = CLIPTokenizer.from_pretrained(self.load_pretrain_path, subfolder="tokenizer")
        concept_tokens = [self.token_templates.format(id) for id in range(self.num_classes)]
        tokenizer.add_tokens(concept_tokens)

        categories = [[t.strip() for t in c.strip().split(',')] for c in self.categories]
        categories = [[d.split(' ')[-1] for d in c] for c in categories]
        meta_tokens = [select_meta_tokens(c, tokenizer)[0] for c in categories]

        caption_ids_meta_token = [[tokenizer(
            template.format(token),
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0] for template in self.caption_templates] for token in meta_tokens]
        caption_ids_concept_token = [[tokenizer(
            template.format(token),
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0] for template in self.caption_templates] for token in concept_tokens]

        cat2tokens = [dict(meta_token=a, concept_token=b, caption_ids_meta_token=c, caption_ids_concept_token=d)
                      for a, b, c, d in
                      zip(meta_tokens, concept_tokens, caption_ids_meta_token, caption_ids_concept_token)]

        # format tokens by sample
        load_gt = True
        sample2tokens = [cat2tokens[id] for id in self.labels]

        samples = []
        for idx, lt in enumerate(sample2tokens):
            scores = []
            ids_concept_token = []
            ids_meta_token = []
            if self.pred_logits is not None:
                tensor = torch.Tensor(self.pred_logits[idx]).topk(5)
                tmp = [cat2tokens[id] for id in tensor.indices.numpy().tolist()]
                scores.extend([float(s) for s in tensor.values.numpy().tolist()])
                ids_concept_token.extend([t['caption_ids_concept_token'] for t in tmp])
                ids_meta_token.extend([t['caption_ids_meta_token'] for t in tmp])
            if load_gt:
                scores.append(0.0)
                ids_concept_token.append(lt['caption_ids_concept_token'])
                ids_meta_token.append(lt['caption_ids_meta_token'])
            samples.append(dict(
                scores=scores,
                caption_ids_concept_token=ids_concept_token,
                caption_ids_meta_token=ids_meta_token,
            ))
        
        self.tokenizer = tokenizer
        self.cat2tokens = cat2tokens
        self.samples = samples

    def init_text_encoder(self, text_encoder):
        text_encoder.resize_token_embeddings(len(self.tokenizer))
        if self.test_mode or (self.load_token_path is not None):
            text_encoder = self.load_embeddings(text_encoder)
        elif not self.test_mode:
            text_encoder = self.init_embeddings(text_encoder)
        return text_encoder

    def load_embeddings(self, text_encoder):
        missing_token = False
        token_embeds = text_encoder.get_input_embeddings().weight.data.clone()
        for token in self.cat2tokens:
            concept_token = token['concept_token']
            concept_token_id = self.tokenizer.encode(concept_token, add_special_tokens=False)[-1]
            token_bin = os.path.join(self.load_token_path, f"{concept_token_id}.bin")
            if not os.path.exists(token_bin):
                missing_token = True
                continue
            token_embeds[concept_token_id] = torch.load(token_bin)[concept_token]
        text_encoder.get_input_embeddings().weight = torch.nn.Parameter(token_embeds)
        if missing_token:
            print(f"WARN: {self.__class__.__name__}:\t missing token.", flush=True)
        return text_encoder

    def init_embeddings(self, text_encoder):
        token_embeds = text_encoder.get_input_embeddings().weight.data.clone()
        for token in self.cat2tokens:
            meta_token_id = self.tokenizer.encode(token['meta_token'], add_special_tokens=False)[0]
            concept_token_id = self.tokenizer.encode(token['concept_token'], add_special_tokens=False)[0]
            token_embeds[concept_token_id] = token_embeds[meta_token_id]
        text_encoder.get_input_embeddings().weight = torch.nn.Parameter(token_embeds)
        return text_encoder

    def filter_classes(self):
        if isinstance(self.keep_class, int):
            mask = np.array(self.labels) == self.keep_class
        else:
            mask = np.zeros_like(self.labels)
            for cls in self.keep_class:
                mask = mask + (np.array(self.labels) == cls).astype(float)
            mask = mask.astype(bool)

        self.labels = [el for i, el in enumerate(self.labels) if mask[i]]
        if self.names is not None and len(self.names) != 0:
            self.names = [el for i, el in enumerate(self.names) if mask[i]]
        if self.bboxes is not None and len(self.bboxes) != 0:
            self.bboxes = [el for i, el in enumerate(self.bboxes) if mask[i]]

    def init_train_pipelines(self, **kwargs):
        center_crop = kwargs.pop("center_crop", False)
        resize_size = kwargs.pop("resize_size", 512)
        crop_size = kwargs.pop("crop_size", 512)
        brightness = kwargs.pop("brightness", 0.1)
        contrast = kwargs.pop("contrast", 0.1)
        saturation = kwargs.pop("saturation", 0.1)
        hue = kwargs.pop("hue", 0.1)
        interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[kwargs.pop("interpolation", "bicubic")]
        
        train_transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop((crop_size, crop_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness, contrast, saturation, hue)
            ])
        
        def train_pipeline(data):
            img = data['img']
            if center_crop:
                crop = min(img.shape[0], img.shape[1])
                h, w, = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
            img = Image.fromarray(img)
            img = train_transform(img)
            img = img.resize((crop_size, crop_size), resample=interpolation)
            img = np.array(img).astype(np.uint8)
            img = (img / 127.5 - 1.0).astype(np.float32)
            data["img"] = torch.from_numpy(img).permute(2, 0, 1)
            return data
        
        return train_pipeline

    def init_test_pipelines(self, **kwargs):
        crop_size = kwargs.pop("crop_size", 512)
        interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[kwargs.pop("interpolation", "bicubic")]

        def test_pipeline(data):
            img = data['img']
            bbox = data['gt_bboxes']
            image_width, image_height = data['ori_shape']
            img = Image.fromarray(img)
            img = img.resize((crop_size, crop_size), resample=interpolation)
            img = np.array(img).astype(np.uint8)
            img = (img / 127.5 - 1.0).astype(np.float32)
            data["img"] = torch.from_numpy(img).permute(2, 0, 1)

            [x1, y1, x2, y2] = np.split(bbox, 4, 1)
            resize_size = crop_size
            shift_size = 0
            left_bottom_x = np.maximum(x1 / image_width * resize_size - shift_size, 0).astype(int)
            left_bottom_y = np.maximum(y1 / image_height * resize_size - shift_size, 0).astype(int)
            right_top_x = np.minimum(x2 / image_width * resize_size - shift_size, crop_size - 1).astype(int)
            right_top_y = np.minimum(y2 / image_height * resize_size - shift_size, crop_size - 1).astype(int)

            gt_bbox = np.concatenate((left_bottom_x, left_bottom_y, right_top_x, right_top_y), axis=1).reshape(-1)
            gt_bbox = " ".join(list(map(str, gt_bbox)))
            data['gt_bboxes'] = gt_bbox

            
            return data
        
        return test_pipeline

    def prepare_data(self, idx):
        pipelines = self.train_pipelines if not self.test_mode else self.test_pipelines
        bboxes = [] if not self.test_mode else self.bboxes[idx]

        name = self.names[idx]
        label = self.labels[idx]
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image_size = list(image.size)
        img = np.array(image).astype(np.uint8)
        data = pipelines(dict(
            img=img, ori_shape=image_size,
            gt_labels=label, gt_names=name, gt_bboxes=bboxes,
        ))
        
        data = dict(
            img=data["img"],
            ori_shape=data["ori_shape"],
            gt_labels=data["gt_labels"],
            gt_bboxes=data["gt_bboxes"],
            name=data["gt_names"],
        )

        if self.test_mode:
            pred_scores = self.pred_logits
            data.update(dict(
                pred_logits=pred_scores[idx],
                pred_top5_ids=pred_scores[idx].topk(5).indices,
            )) 

        return data

    def prepare_tokens(self, idx):
        sample = self.samples[idx]

        if not self.test_mode:
            choice = random.choice(range(len(self.caption_templates)))
            sample = dict(
                caption_ids_concept_token=sample['caption_ids_concept_token'][-1][choice],
                caption_ids_meta_token=sample['caption_ids_meta_token'][-1][choice],
            )

        return sample

    def __len__(self):
        return self.num_images * self.data_repeat

    def __getitem__(self, idx):
        data = self.prepare_data(idx % self.num_images)
        data.update(self.prepare_tokens(idx % self.num_images))
        return data

class SubDataset(BaseDataset):
    def __init__(self, keep_class, dataset):
        self.dataset = dataset

        if isinstance(keep_class, int):
            keep_class = [keep_class]
        self.keep_class = keep_class

        if isinstance(self.keep_class, int):
            mask = np.array(self.dataset.labels) == self.keep_class
        else:
            mask = np.zeros_like(self.dataset.labels)
            for cls in self.keep_class:
                mask = mask + (np.array(self.dataset.labels) == cls).astype(float)
            mask = mask.astype(bool)

        self.names = self.dataset.names
        self.image_paths = self.dataset.image_paths
        self.data_repeat = self.dataset.data_repeat
        self.labels = self.dataset.labels
        self.samples = self.dataset.samples
        self.caption_templates = self.dataset.caption_templates
        self.test_mode = self.dataset.test_mode
        self.num_images = self.dataset.num_images
        self.train_pipelines = self.dataset.train_pipelines
        self.test_pipelines = self.dataset.test_pipelines

        self.names = [el for i, el in enumerate(self.names) if mask[i]]
        self.image_paths = [el for i, el in enumerate(self.image_paths) if mask[i]]
        self.labels = [el for i, el in enumerate(self.labels) if mask[i]]
        self.samples = [el for i, el in enumerate(self.samples) if mask[i]]
        self.num_images = len(self.image_paths)

class ImagenetDataset(BaseDataset):
    def __init__(self,
                 root=".",
                 repeats=1,
                 crop_size=512,
                 resize_size=512,
                 test_mode=False,
                 keep_class=None,
                 center_crop=False,
                 text_encoder=None,
                 load_class_path=None,
                 load_token_path=None,
                 load_pretrain_path=None,
                 interpolation="bicubic",
                 token_templates="<imagenet-{}>",
                 caption_templates=caption_templates,
                 **kwargs,
                 ):
        super().__init__(root=root,
                         repeats=repeats,
                         crop_size=crop_size,
                         resize_size=resize_size,
                         test_mode=test_mode,
                         keep_class=keep_class,
                         center_crop=center_crop,
                         text_encoder=text_encoder,
                         load_class_path=load_class_path,
                         load_token_path=load_token_path,
                         load_pretrain_path=load_pretrain_path,
                         interpolation=interpolation,
                         token_templates=token_templates,
                         caption_templates=caption_templates,
                         **kwargs)

    def load_data(self):
        """
        returns:
            self.names: []
            self.labels: []
            self.bboxes: []; (x1,y1,x2,y2)
            self.pred_logits: [] | None
            self.image_paths: []
        """
        class_file = os.path.join(self.root, 'ILSVRC2012_list', 'LOC_synset_mapping.txt')
        self.categories = []
        with open(class_file, 'r') as f:
            discriptions = f.readlines()  # "n01882714 koala..."
            for id, line in enumerate(discriptions):
                tag, description = line.strip().split(' ', maxsplit=1)
                self.categories.append(description)
        self.num_classes = len(self.categories)

        self.names = []
        self.labels = []
        self.bboxes = []
        self.pred_logits = []
        self.image_paths = []

        data_file = os.path.join(self.root, 'ILSVRC2012_list', 'train.txt')
        image_dir = os.path.join(self.root, 'train')
        if self.test_mode:
            data_file = os.path.join(self.root, 'ILSVRC2012_list', 'val_folder_new.txt')
            image_dir = os.path.join(self.root, 'val')

        with open(data_file) as f:
            datamappings = f.readlines() # "n01440764/n01440764_10026.JEPG 0"
            for id, line in enumerate(datamappings):
                info = line.strip().split()
                self.names.append(info[0][:-5]) # "n01440764/n01440764_10026"
                self.labels.append(int(info[1])) # "0"
                if self.test_mode:
                    self.bboxes.append(np.array(list(map(float, info[2:]))).reshape(-1, 4))
        if self.keep_class is not None:
            self.filter_classes()
        self.pred_logits = None
        if self.test_mode:
            with open(self.load_class_path, 'r') as f:
                name2result = json.load(f)
                self.pred_logits = [torch.Tensor(name2result[name]['pred_scores']) for name in self.names]
        self.image_paths = [os.path.join(image_dir, name + '.JPEG') for name in self.names]
        self.num_images = len(self.labels)

class CUBDataset(BaseDataset):
    def __init__(self,
                 root=".",
                 repeats=1,
                 crop_size=512,
                 resize_size=512,
                 test_mode=False,
                 keep_class=None,
                 center_crop=False,
                 text_encoder=None,
                 load_class_path=None,
                 load_token_path=None,
                 load_pretrain_path=None,
                 interpolation="bicubic",
                 token_templates="<cub-{}>",
                 caption_templates=caption_templates,
                 **kwargs,
                 ):
        super().__init__(root=root,
                         repeats=repeats,
                         crop_size=crop_size,
                         resize_size=resize_size,
                         test_mode=test_mode,
                         keep_class=keep_class,
                         center_crop=center_crop,
                         text_encoder=text_encoder,
                         load_class_path=load_class_path,
                         load_token_path=load_token_path,
                         load_pretrain_path=load_pretrain_path,
                         interpolation=interpolation,
                         token_templates=token_templates,
                         caption_templates=caption_templates,
                         **kwargs)

    def load_data(self):
        self.categories = ["bird"] * 200
        self.num_classes = len(self.categories)

        images_file = os.path.join(self.root, 'images.txt')
        labels_file = os.path.join(self.root, 'image_class_labels.txt')
        splits_file = os.path.join(self.root, 'train_test_split.txt')
        bboxes_file = os.path.join(self.root, 'bounding_boxes.txt')

        with open(images_file, 'r') as f:
            lines = f.readlines()
            image_list = [line.strip().split(' ')[1] for line in lines]
        with open(labels_file, 'r') as f:
            lines = f.readlines()
            labels_list = [line.strip().split(' ')[1] for line in lines]
        with open(splits_file, 'r') as f:
            lines = f.readlines()
            splits_list = [line.strip().split(' ')[1] for line in lines]
        with open(bboxes_file, 'r') as f:
            lines = f.readlines()
            bboxes_list = [line.strip().split(' ')[1:] for line in lines]

        train_index = [i for i, v in enumerate(splits_list) if v == '1']
        test_index = [i for i, v in enumerate(splits_list) if v == '0']
        index_list = train_index if not self.test_mode else test_index
        self.names = [image_list[i] for i in index_list]
        self.labels = [int(labels_list[i]) - 1 for i in index_list]
        self.bboxes = [np.array(list(map(float, bboxes_list[i]))).reshape(-1, 4)
                       for i in index_list]
        for i in range(len(self.bboxes)):
            self.bboxes[i][:, 2:4] = self.bboxes[i][:, 0:2] + self.bboxes[i][:, 2:4]

        if self.keep_class is not None:
            self.filter_classes()
        self.pred_logits = None
        if self.test_mode:
            with open(self.load_class_path, 'r') as f:
                name2result = json.load(f)
                self.pred_logits = [torch.Tensor(name2result[name]['pred_scores']) for name in self.names]
        self.image_paths = [os.path.join(self.root, 'images', name) for name in self.names]
        self.num_images = len(self.labels)


