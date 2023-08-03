import os
import sys
import math
import yaml
import argparse
import tqdm
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

from models.attn import AttentionStore
from datasets.base import CUBDataset, ImagenetDataset, SubDataset
from datasets.evaluation import Evaluator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = dict(cub=CUBDataset, imagenet=ImagenetDataset)
OPTIMIZER = dict(AdamW=torch.optim.AdamW)

def set_env(benchmark=True):
    # get config
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--function", type=str, default="test", required=True)
    parser.add_argument("--config", type=str, default="configs/cub.yml", help="Config file", required=True)
    parser.add_argument("--opt", type=str, default="dict()", help="Override options.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    def load_yaml_conf(config):
        def dict_fix(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    v = dict_fix(v)
                elif v == "None":
                    v = None
                elif v == "False":
                    v = False
                elif v == "True":
                    v = True
                elif isinstance(v, str):
                    v = float(v) if v.isdigit() else v
                d[k] = v
            return d

        assert os.path.exists(config), "ERROR: no config file found."
        with open(config, "r") as f:
            config = yaml.safe_load(f)
        config = dict_fix(config)
        return config

    def format_conf(config):
        if config["train"]["max_train_steps"] is None:
            config["train"]["max_train_steps"] = 0

        keep_class = config["data"]["keep_class"]
        if keep_class is not None:
            if isinstance(keep_class, int):
                keep_class = [keep_class]
            elif isinstance(keep_class, list) and len(keep_class)==2:
                keep_class = list(range(keep_class[0], keep_class[1]+1))
            else:
                assert isinstance(keep_class, list)

        data = dict(
            train=dict(
                batch_size=config["train"]["batch_size"],
                shuffle=True,
                dataset=dict(type=config["data"]["dataset"],
                             root=config["data"]["root"],
                             keep_class=keep_class,
                             crop_size=config["data"]["crop_size"],
                             resize_size=config["data"]["resize_size"],
                             load_pretrain_path=config["train"]["load_pretrain_path"],
                             load_token_path=config["train"]["load_token_path"],
                             save_path=config["train"]["save_path"],
                             ),
            ),
            test=dict(
                batch_size=config["test"]["batch_size"],
                shuffle=False,
                dataset=dict(type=config["data"]["dataset"],
                             root=config["data"]["root"],
                             keep_class=keep_class,
                             crop_size=config["data"]["crop_size"],
                             resize_size=config["data"]["resize_size"],
                             load_pretrain_path=config["test"]["load_pretrain_path"],
                             load_class_path=config["test"]["load_class_path"],
                             load_token_path=config["test"]["load_token_path"],
                             ),
            ),
        )

        optimizer = dict(
            type="AdamW",
            lr=config["train"]["learning_rate"],
            betas=(config["train"]["adam_beta1"], config["train"]["adam_beta2"]),
            weight_decay=eval(config["train"]["adam_weight_decay"]),
            eps=eval(config["train"]["adam_epsilon"]),
        )

        lr_scheduler = dict(
            type=config["train"]["lr_scheduler"],
            num_warmup_steps=config["train"]["lr_warmup_steps"] * config["train"]["gradient_accumulation_steps"],
            num_training_steps=config["train"]["max_train_steps"] * config["train"]["gradient_accumulation_steps"],
        )

        accelerator = dict(
            # logging_dir=os.path.join(config["train"]["save_path"], "logs"),
            gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
            mixed_precision="no",
            log_with=None
        )

        model = dict(

        )

        train = dict(
            epochs=config["train"]["num_train_epochs"],
            scale_lr=config["train"]["scale_lr"],
            push_to_hub=False,
            save_path=config["train"]["save_path"],
            save_step=config["train"]["save_steps"],
            load_pretrain_path=config["train"]["load_pretrain_path"],
            load_token_path=config["train"]["load_token_path"],
        )

        test = dict(
            cam_thr=config["test"]["cam_thr"],
            eval_mode=config["test"]["eval_mode"],
            combine_ratio=config["test"]["combine_ratio"],
            load_pretrain_path=config["test"]["load_pretrain_path"],
            load_token_path=config["test"]["load_token_path"],
            load_unet_path=config["test"]["load_unet_path"] if config["test"]["load_unet_path"] is not None else os.path.join(config["test"]["load_pretrain_path"], "unet"),
            save_vis_path=config["test"]["save_vis_path"],
            save_log_path=config["test"]["save_log_path"]
        )

        config = dict(
            model=model,
            data=data,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            train=train,
            test=test,
        )

        return config

    def merge_conf(config, extra_config):
        for key, value in extra_config.items():
            if not isinstance(value, dict):
                config[key] = value
            else:
                merge_value = merge_conf(config.get(key, dict()), value)
                config[key] = merge_value
        return config

    # parse config file
    config = load_yaml_conf(args.config)

    # override options
    extra_config = eval(args.opt)
    merge_conf(config, extra_config)

    config = format_conf(config)

    # set random seed
    if args.seed is not None:
        set_seed(args.seed, device_specific=False)
    
    # set benchmark
    torch.backends.cudnn.benchmark = benchmark
    
    return args, config

def test(config):
    split = "test"
    device = 'cuda'
    torch_dtype = torch.float16

    eval_mode = config["test"]["eval_mode"]
    load_pretrain_path = config["test"]["load_pretrain_path"]
    keep_class = config["data"][split]["dataset"]["keep_class"]
    combine_ratio = config["test"]["combine_ratio"]
    save_log_path = config["test"]["save_log_path"]
    load_unet_path = config["test"]["load_unet_path"]
    batch_size = config["data"][split]["batch_size"]

    text_encoder = CLIPTextModel.from_pretrained(load_pretrain_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(load_pretrain_path, subfolder="vae", torch_dtype=torch_dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(load_unet_path, torch_dtype=torch_dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(load_pretrain_path, subfolder="scheduler")

    data_configs = config["data"][split].copy()
    dataset_config = data_configs.pop("dataset", None)
    dataset_type = dataset_config.pop("type", "imagenet")
    dataset_config.update(dict(
        test_mode=(split == "val" or split == "test"),
        text_encoder=text_encoder))
    dataset = DATASETS[dataset_type](**dataset_config)
    dataloader = torch.utils.data.DataLoader(dataset, **data_configs)

    vae.eval()
    unet.eval()
    text_encoder.eval()

    evaluator = Evaluator(logfile=save_log_path, len_dataloader=len(dataloader))
    controller = AttentionStore(batch_size=batch_size)
    AttentionStore.register_attention_control(controller, unet)

    if keep_class is None:
        keep_class = list(range(dataset.num_classes))
    print(f"INFO: Test Save:\t [log: {str(config['test']['save_log_path'])}] [vis: {str(config['test']['save_vis_path'])}]", flush=True)
    print(f"INFO: Test CheckPoint:\t [token: {str(config['test']['load_token_path'])}] [unet: {str(config['test']['load_unet_path'])}]", flush=True)
    print(f"INFO: Test Class [{keep_class[0]}-{keep_class[-1]}]:\t [dataset: {dataset_type}] [eval mode: {eval_mode}] "
          f"[cam thr: {config['test']['cam_thr']}] [combine ratio: {combine_ratio}]", flush=True)

    for step, data in enumerate(tqdm.tqdm(dataloader)):
        if eval_mode == "gtk":
            image = data["img"].to(torch_dtype).to(device)
            latents = vae.encode(image).latent_dist.sample().detach() * 0.18215
            noise = torch.randn(latents.shape).to(latents.device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
            ).long()

            representative_embeddings = [text_encoder(ids.to(device))[0] for ids in data["caption_ids_concept_token"][-1]]
            representative_embeddings = sum(representative_embeddings)/len(data["caption_ids_concept_token"][-1])

            discriminative_embeddings = [text_encoder(ids.to(device))[0] for ids in data["caption_ids_meta_token"][-1]]
            discriminative_embeddings = sum(discriminative_embeddings) / len(data["caption_ids_meta_token"][-1])
            combine_embeddings = combine_ratio * representative_embeddings + (1-combine_ratio) * discriminative_embeddings
            combine_embeddings = combine_embeddings.to(torch_dtype)

            for t in [0, 99]:
                timesteps = torch.ones_like(timesteps) * t
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(torch_dtype)
                noise_pred = unet(noisy_latents, timesteps, combine_embeddings).sample

            cams = controller.diffusion_cam(idx=5)

            controller.reset()

            cams_tensor = torch.from_numpy(cams).to(device).unsqueeze(dim=1)
            pad_cams = cams_tensor.repeat(1, dataset.num_classes, 1, 1)

            evaluator(data["img"], data["gt_labels"], data['gt_bboxes'], data["pred_logits"], pad_cams, data["name"], config, step)
        elif eval_mode == "top1":
            cams_all = []

            image = data["img"].to(torch_dtype).to(device)
            latents = vae.encode(image).latent_dist.sample().detach() * 0.18215
            noise = torch.randn(latents.shape).to(latents.device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
            ).long()

            for top_idx in [0, 5]:

                # save inference cost
                if top_idx == 5:
                    top1_idx = data["pred_top5_ids"][:, 0]
                    gtk_idx = torch.LongTensor(data["gt_labels"]).to(torch.int64)
                    if torch.all(top1_idx == gtk_idx):
                        cams_all = cams_all*2
                        break

                representative_embeddings = [text_encoder(ids.to(device))[0] for ids in
                                             data["caption_ids_concept_token"][top_idx]]
                representative_embeddings = sum(representative_embeddings) / len(data["caption_ids_concept_token"][top_idx])

                discriminative_embeddings = [text_encoder(ids.to(device))[0] for ids in
                                             data["caption_ids_meta_token"][top_idx]]
                discriminative_embeddings = sum(discriminative_embeddings) / len(data["caption_ids_meta_token"][top_idx])
                combine_embeddings = combine_ratio * representative_embeddings + (
                            1 - combine_ratio) * discriminative_embeddings
                combine_embeddings = combine_embeddings.to(torch_dtype)

                for t in [0, 99]:
                    timesteps = torch.ones_like(timesteps) * t
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(torch_dtype)
                    noise_pred = unet(noisy_latents, timesteps, combine_embeddings).sample

                controller.batch_size = len(noise_pred)
                cams = controller.diffusion_cam(idx=5)
                controller.reset()
                cams_all.append(cams)

            cams_tensor = torch.from_numpy(cams_all[0]).to(device).unsqueeze(dim=1)
            pad_cams = cams_tensor.repeat(1, dataset.num_classes, 1, 1)
            for i, pad_cam in enumerate(pad_cams):
                pad_cam[data["gt_labels"][i]] = torch.from_numpy(cams_all[-1])[i]

            evaluator(data["img"], data["gt_labels"], data['gt_bboxes'], data["pred_logits"], pad_cams, data["name"], config, step)
        elif eval_mode == "top5":
            cams_all = []

            image = data["img"].to(torch_dtype).to(device)
            latents = vae.encode(image).latent_dist.sample().detach() * 0.18215
            noise = torch.randn(latents.shape).to(latents.device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
            ).long()

            for top_idx in range(6):
                representative_embeddings = [text_encoder(ids.to(device))[0] for ids in
                                             data["caption_ids_concept_token"][top_idx]]
                representative_embeddings = sum(representative_embeddings) / len(data["caption_ids_concept_token"][top_idx])

                discriminative_embeddings = [text_encoder(ids.to(device))[0] for ids in
                                             data["caption_ids_meta_token"][top_idx]]
                discriminative_embeddings = sum(discriminative_embeddings) / len(data["caption_ids_meta_token"][top_idx])
                combine_embeddings = combine_ratio * representative_embeddings + (
                            1 - combine_ratio) * discriminative_embeddings
                combine_embeddings = combine_embeddings.to(torch_dtype)

                for t in [0, 99]:
                    timesteps = torch.ones_like(timesteps) * t
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(torch_dtype)
                    noise_pred = unet(noisy_latents, timesteps, combine_embeddings).sample

                cams = controller.diffusion_cam(idx=5)
                controller.reset()
                cams_all.append(cams)

            cams = torch.from_numpy(np.stack([cam for cam in cams_all[:-1]], 1))
            pad_cams = torch.zeros((batch_size, dataset.num_classes, *cams.shape[-2:]))
            for i, pad_cam in enumerate(pad_cams):
                pad_cam[data["pred_top5_ids"][i]] = cams[i]
                pad_cam[data["gt_labels"][i]] = torch.from_numpy(cams_all[-1])[i]

            evaluator(data["img"], data["gt_labels"], data['gt_bboxes'], data["pred_logits"], pad_cams, data["name"], config, step)
        else:
            raise ValueError("select eval_mode in [gtk, top1, top5].")

def train_token(config):
    split = "train"
    device = 'cuda'
    torch_dtype = torch.float32

    load_pretrain_path = config["train"]["load_pretrain_path"]
    keep_class = config["data"][split]["dataset"]["keep_class"]

    text_encoder = CLIPTextModel.from_pretrained(load_pretrain_path, subfolder="text_encoder", torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(load_pretrain_path, subfolder="vae", torch_dtype=torch_dtype)
    unet = UNet2DConditionModel.from_pretrained(load_pretrain_path, subfolder="unet", torch_dtype=torch_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(load_pretrain_path, subfolder="scheduler")

    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    freeze_params(itertools.chain(
        vae.parameters(),
        unet.parameters(),
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    ))

    data_configs = config["data"][split].copy()
    dataset_config = data_configs.pop("dataset", None)
    dataset_type = dataset_config.pop("type", "imagenet")
    dataset_config.update(dict(
        test_mode=(split == "val" or split == "test"),
        text_encoder=text_encoder))
    dataset = DATASETS[dataset_type](**dataset_config)

    def train_loop(config, class_id, text_encoder=None, unet=None, vae=None, dataset=None):

        def get_grads_to_zero(class_id, dataset):
            tokenizer = dataset.tokenizer
            index_grads_to_zero = torch.ones((len(tokenizer))).bool()
            concept_token = dataset.cat2tokens[class_id]["concept_token"]
            token_id = tokenizer.encode(concept_token, add_special_tokens=False)[0]
            index_grads_to_zero[token_id] = False
            return index_grads_to_zero, token_id

        index_grads_to_zero, token_id = get_grads_to_zero(class_id, dataset)
        config = copy.deepcopy(config)

        subdataset = SubDataset(keep_class=[class_id], dataset=dataset)
        data_configs = config["data"][split].copy()
        data_configs.pop("dataset", None)
        dataloader = torch.utils.data.DataLoader(subdataset, **data_configs)

        accelerator_config = config.get("accelerator", None)
        accelerator = Accelerator(**accelerator_config)
        if accelerator.is_main_process:
            accelerator.init_trackers("wsol", config=config)

        save_path = config['train']['save_path']
        batch_size = config['data']['train']['batch_size']
        gradient_accumulation_steps = config['accelerator']['gradient_accumulation_steps']
        num_train_epochs = config['train']['epochs']
        max_train_steps = config['lr_scheduler']['num_training_steps'] // gradient_accumulation_steps
        total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps

        if config['train']['scale_lr']:
            config['optimizer']['lr'] = config['optimizer']['lr'] * total_batch_size
        if (max_train_steps is None) or (max_train_steps == 0):
            num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
            max_train_steps = num_train_epochs * num_update_steps_per_epoch // accelerator.num_processes

        config['lr_scheduler']['num_training_steps'] = max_train_steps * gradient_accumulation_steps
        optimizer_config = config.get("optimizer", None)
        optimizer_type = optimizer_config.pop("type", "AdamW")
        lr_scheduler_config = config.get("lr_scheduler", None)
        lr_scheduler_type = lr_scheduler_config.pop("type", "constant")

        optimizer = OPTIMIZER[optimizer_type](text_encoder.get_input_embeddings().parameters(), **optimizer_config)
        lr_scheduler = get_scheduler(name=lr_scheduler_type, optimizer=optimizer, **lr_scheduler_config)

        if accelerator.is_main_process:
            print(f"INFO: Train Save:\t [ckpt: {save_path}]", flush=True)
            print(f"INFO: Train Class [{class_id}]:\t [num samples: {len(dataloader)}] "
                  f"[num epochs: {num_train_epochs}] [batch size: {total_batch_size}] "
                  f"[total steps: {max_train_steps}]", flush=True)

        vae, unet, text_encoder, optimizer, lr_scheduler, dataloader = accelerator.prepare(vae, unet, text_encoder,
                                                                                           optimizer, lr_scheduler,
                                                                                           dataloader)
        vae.eval()
        unet.eval()

        global_step = 0
        progress_bar = tqdm.tqdm(range(max_train_steps), disable=(not accelerator.is_local_main_process))
        for epoch in range(num_train_epochs):
            text_encoder.train()
            progress_bar.set_description(f"Epoch[{epoch+1}/{num_train_epochs}] ")
            for step, data in enumerate(dataloader):
                with accelerator.accumulate(text_encoder):
                    combine_embeddings = text_encoder(data["caption_ids_concept_token"])[0]

                    image = data["img"].to(torch_dtype)  # use torch.float16 rather than float32
                    latents = vae.encode(image).latent_dist.sample().detach() * 0.18215

                    noise = torch.randn(latents.shape, device=latents.device, dtype=torch_dtype)
                    timesteps = torch.randint(low=0, high=noise_scheduler.config.num_train_timesteps,
                                              size=(latents.shape[0],), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    noise_pred = unet(noisy_latents, timesteps, combine_embeddings).sample
                    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                    accelerator.backward(loss)

                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(refresh=False, **logs)
                    accelerator.log(logs, step=global_step)

                    if global_step >= max_train_steps:
                        break

                if global_step >= max_train_steps:
                    break

            # save concept token embeddings per epoch
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                out_dir = os.path.join(save_path, "tokens")
                if epoch != num_train_epochs - 1:
                    out_dir = os.path.join(save_path, f"tokens_e{epoch}")
                os.makedirs(out_dir, exist_ok=True)
                unwrap_text_encoder = accelerator.unwrap_model(text_encoder)
                concept_token = dataset.tokenizer.decode(token_id)
                concept_token_embeddings = unwrap_text_encoder.get_input_embeddings().weight[token_id]
                dct = {concept_token: concept_token_embeddings.detach().cpu()}
                torch.save(dct, os.path.join(out_dir, f"{token_id}.bin"))

        accelerator.end_training()

    if keep_class is None:
        keep_class = list(range(dataset.num_classes))
    for class_id in keep_class:
        train_loop(config, class_id, text_encoder, unet, vae, dataset)

def train_unet(config):
    split = "train"
    device = 'cuda'
    torch_dtype = torch.float32

    load_pretrain_path = config["train"]["load_pretrain_path"]

    text_encoder = CLIPTextModel.from_pretrained(load_pretrain_path, subfolder="text_encoder", torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(load_pretrain_path, subfolder="vae", torch_dtype=torch_dtype)
    unet = UNet2DConditionModel.from_pretrained(load_pretrain_path, subfolder="unet", torch_dtype=torch_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(load_pretrain_path, subfolder="scheduler")

    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    freeze_params(itertools.chain(
        vae.parameters(),
        text_encoder.parameters(),
    ))

    data_configs = config["data"][split].copy()
    dataset_config = data_configs.pop("dataset", None)
    dataset_type = dataset_config.pop("type", "imagenet")
    dataset_config.update(dict(
        test_mode=(split == "val" or split == "test"),
        text_encoder=text_encoder))
    dataset = DATASETS[dataset_type](**dataset_config)
    dataloader = torch.utils.data.DataLoader(dataset, **data_configs)

    def train_loop(config, text_encoder=None, unet=None, vae=None, dataloader=None):
        config = copy.deepcopy(config)

        accelerator_config = config.get("accelerator", None)
        accelerator = Accelerator(**accelerator_config)
        if accelerator.is_main_process:
            accelerator.init_trackers("wsol", config=config)

        save_path = config['train']['save_path']
        save_step = config['train']['save_step']
        batch_size = config['data']['train']['batch_size']
        gradient_accumulation_steps = config['accelerator']['gradient_accumulation_steps']
        num_train_epochs = config['train']['epochs']
        max_train_steps = config['lr_scheduler']['num_training_steps'] // gradient_accumulation_steps
        total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps

        if config['train']['scale_lr']:
            config['optimizer']['lr'] = config['optimizer']['lr'] * total_batch_size
        if (max_train_steps is None) or (max_train_steps == 0):
            num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
            max_train_steps = num_train_epochs * num_update_steps_per_epoch // accelerator.num_processes

        config['lr_scheduler']['num_training_steps'] = max_train_steps * gradient_accumulation_steps
        optimizer_config = config.get("optimizer", None)
        optimizer_type = optimizer_config.pop("type", "AdamW")
        lr_scheduler_config = config.get("lr_scheduler", None)
        lr_scheduler_type = lr_scheduler_config.pop("type", "constant")

        optimizer = OPTIMIZER[optimizer_type](accelerator.unwrap_model(unet).parameters(), **optimizer_config)
        lr_scheduler = get_scheduler(name=lr_scheduler_type, optimizer=optimizer, **lr_scheduler_config)

        if accelerator.is_main_process:
            print(f"INFO: Train Save:\t [ckpt: {save_path}]", flush=True)
            print(f"INFO: Finetune UNet:\t [num samples: {len(dataloader)}] "
                  f"[num epochs: {num_train_epochs}] [batch size: {total_batch_size}] "
                  f"[total steps: {max_train_steps}] [save step: {save_step}]", flush=True)

        vae, unet, text_encoder, optimizer, lr_scheduler, dataloader = accelerator.prepare(vae, unet, text_encoder,
                                                                                           optimizer, lr_scheduler,
                                                                                           dataloader)
        vae.eval()
        text_encoder.eval()

        global_step = 0
        progress_bar = tqdm.tqdm(range(max_train_steps), disable=(not accelerator.is_local_main_process))
        for epoch in range(num_train_epochs):
            unet.train()
            progress_bar.set_description(f"Epoch[{epoch + 1}/{num_train_epochs}] ")
            for step, data in enumerate(dataloader):
                with accelerator.accumulate(unet):
                    representative_embeddings = text_encoder(data["caption_ids_concept_token"])[0]
                    discriminative_embeddings = text_encoder(data["caption_ids_meta_token"])[0]
                    combine_embeddings = 0.5 * representative_embeddings + 0.5 * discriminative_embeddings

                    image = data["img"].to(torch_dtype)
                    latents = vae.encode(image).latent_dist.sample().detach() * 0.18215

                    noise = torch.randn(latents.shape, device=latents.device, dtype=torch_dtype)
                    timesteps = torch.randint(low=0, high=noise_scheduler.config.num_train_timesteps,
                                              size=(latents.shape[0],), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    noise_pred = unet(noisy_latents, timesteps, combine_embeddings).sample
                    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                    accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(refresh=False, **logs)
                    accelerator.log(logs, step=global_step)

                    if global_step >= max_train_steps:
                        break
                    elif (global_step + 1) % save_step == 0:
                        if accelerator.sync_gradients:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                out_dir = os.path.join(save_path, f"unet_s{global_step}")
                                os.makedirs(out_dir, exist_ok=True)
                                try:
                                    unet.module.save_pretrained(save_directory=out_dir)
                                except:
                                    unet.save_pretrained(save_directory=out_dir)


                if global_step >= max_train_steps:
                    break

            if accelerator.sync_gradients:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    out_dir = os.path.join(save_path, f"unet_e{global_step}")
                    os.makedirs(out_dir, exist_ok=True)
                    try:
                        unet.module.save_pretrained(save_directory=out_dir)
                    except:
                        unet.save_pretrained(save_directory=out_dir)

        accelerator.end_training()

    train_loop(config, text_encoder, unet, vae, dataloader)

if __name__ == "__main__":
    args, config = set_env(benchmark=True)
    print(args)
    eval(args.function)(config)


