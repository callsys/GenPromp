import cv2
import torch
import numpy as np
import torch.utils.checkpoint
import matplotlib.pyplot as plt
import einops

class AttentionStore():
    def __init__(self, batch_size=2):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.active = True
        self.batch_size = batch_size

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    
    @property
    def num_uncond_att_layers(self):
        return 0

    def step_callback(self, x_t):
        return x_t

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if self.active:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if self.active:
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    @staticmethod
    def register_attention_control(controller, model):

        def ca_attention(self, place_in_unet):

            def get_attention_scores(query, key, attention_mask=None):
                dtype = query.dtype
                
                if self.upcast_attention:
                    query = query.float()
                    key = key.float()

                if attention_mask is None:
                    baddbmm_input = torch.empty(
                        query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
                    )
                    beta = 0
                else:
                    baddbmm_input = attention_mask
                    beta = 1

                attention_scores = torch.baddbmm(
                    baddbmm_input,
                    query,
                    key.transpose(-1, -2),
                    beta=beta,
                    alpha=self.scale,
                )

                if self.upcast_softmax:
                    attention_scores = attention_scores.float()

                attention_probs = attention_scores.softmax(dim=-1)
                attention_probs = attention_probs.to(dtype)

                if query.shape == key.shape:
                    is_cross = False
                else:
                    is_cross = True

                attention_probs = controller(attention_probs, is_cross, place_in_unet)

                return attention_probs

            return get_attention_scores

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'CrossAttention':
                # net_._attention = ca_attention(net_, place_in_unet)
                net_.get_attention_scores = ca_attention(net_, place_in_unet)
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = model.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
        controller.num_att_layers = cross_att_count

    def aggregate_attention(self, res, from_where, is_cross, bz):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(bz, -1, res, res, item.shape[-1])
                    out.append(cross_maps)
        out = torch.cat(out, dim=1)
        return out.cpu()

    def self_attention_map(self, res, from_where, bz, max_com=10, out_res=64):
        attention_maps = self.aggregate_attention(res, from_where, False, bz)
        maps = []
        for b in range(bz):
            attention_map = attention_maps[b].detach().numpy().astype(np.float32).mean(0).reshape((res**2, res**2))
            u, s, vh = np.linalg.svd(attention_map - np.mean(attention_map, axis=1, keepdims=True))
            images = []
            for i in range(max_com):
                image = vh[i].reshape(res, res)
                # image = image/image.max()
                # image = (image - image.min()) / (image.max() - image.min())
                image = cv2.resize(image, (out_res, out_res), interpolation=cv2.INTER_CUBIC)
                images.append(image)
            map = np.stack(images, 0).max(0)
            maps.append(map)
        return np.stack(maps, 0)

    def cross_attention_map(self, res, from_where, bz, out_res=64, idx=5):
        attention_maps = self.aggregate_attention(res, from_where, True, bz)
        attention_maps = attention_maps[..., idx]
        attention_maps = attention_maps.sum(1) / attention_maps.shape[1]

        maps = []
        for b in range(bz):
            map = attention_maps[b, :, :]
            map = cv2.resize(map.detach().numpy().astype(np.float32), (out_res, out_res),
                                interpolation=cv2.INTER_CUBIC)
            # map = map / map.max()
            maps.append(map)
        return np.stack(maps, 0)

    def diffusion_cam(self, idx=5):
        bz = self.batch_size
        attention_maps_8_ca = self.cross_attention_map(8, ("up", "mid", "down"), bz, idx=idx)
        attention_maps_16_up_ca = self.cross_attention_map(16, ("up",), bz, idx=idx)
        attention_maps_16_down_ca = self.cross_attention_map(16, ("down",), bz, idx=idx)
        attention_maps_ca = (attention_maps_8_ca + attention_maps_16_up_ca + attention_maps_16_down_ca) / 3
        cams = attention_maps_ca
        cams = cams / cams.max((1,2))[:, None, None]
        return cams








