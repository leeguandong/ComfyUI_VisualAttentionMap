import abc
import torch
import numpy as np
import comfy.model_management as mm

from PIL import Image
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from .utils import register_attention_control, AttentionStore, aggregate_attention, \
    text_under_image, convert_preview_image, LOW_RESOURCE


class HFModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["runwayml/stable-diffusion-v1-5"],
                    {"default": "runwayml/stable-diffusion-v1-5"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "TOKENIZER",)
    RETURN_NAMES = ("model", "tokenizer",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "VisualAttn"

    def load_checkpoint(self, model):
        device = mm.get_torch_device()
        model = StableDiffusionPipeline.from_pretrained(model).to(device)
        tokenizer = model.tokenizer
        return (model, tokenizer)


class ShowImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
                {"image": ("IMAGE",)}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "view_images"
    CATEGORY = "VisualAttn"

    def view_images(self, image, num_rows=1, offset_ratio=0.02):
        images = image
        if type(images) is list:
            num_empty = len(images) % num_rows
        elif images.ndim == 4:
            num_empty = images.shape[0] % num_rows
        else:
            images = [images]
            num_empty = 0

        empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
        images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
        num_items = len(images)

        h, w, c = images[0].shape
        offset = int(h * offset_ratio)
        num_cols = num_items // num_rows
        image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                          w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
        for i in range(num_rows):
            for j in range(num_cols):
                image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                    i * num_cols + j]
        pil_img = [Image.fromarray(image_)]
        output_images = convert_preview_image(pil_img)
        return (output_images,)


class Text2ImageInference:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "prompt": ("STRING", {"multiline": True, "default": "A painting of a squirrel eating a burger", }),
            "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
            "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.5, "display": "slider"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
            "optional": {
                "latents": ("LATENTS",)
            }
        }

    RETURN_TYPES = ("VAE", "CONTROLLER", "LATENTS", "STRING")
    RETURN_NAMES = ("vae", "controller", "latents", "prompt")
    FUNCTION = "inference"
    CATEGORY = "VisualAttn"

    def init_latent(self, latent, model, height, width, generator, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, model.unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
        latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
        return latent, latents

    def diffusion_step(self, model, controller, latents, context, t, guidance_scale, low_resource=False):
        if low_resource:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
        else:
            latents_input = torch.cat([latents] * 2) # 40,4,64,64
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
                "sample"]  # latents_input:80,4,64,64;41,77,768
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        latents = controller.step_callback(latents)
        return latents

    def inference(self, model, prompt, steps, guidance_scale, seed, latent=None):
        # import pdb;
        # pdb.set_trace()
        low_resource = LOW_RESOURCE
        prompt = [prompt]

        controller = AttentionStore()
        register_attention_control(model, controller)

        height = width = 512
        batch_size = len(prompt)

        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

        context = [uncond_embeddings, text_embeddings]
        if not low_resource:
            context = torch.cat(context)
        generator = torch.Generator().manual_seed(seed)
        latent, latents = self.init_latent(latent, model, height, width, generator, batch_size)

        # set timesteps
        model.scheduler.set_timesteps(steps)
        for t in tqdm(model.scheduler.timesteps):
            latents = self.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)

        return (model.vae, controller, latents, prompt)


class DecodeLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("VAE",),
            "latents": ("LATENTS",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "VisualAttn"

    def decode(self, vae, latents):
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]
        output_images = convert_preview_image(pil_images)
        return (output_images,)


class ShowCrossAttn:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "controller": ("CONTROLLER",),
            "prompt": ("STRING", {"forceInput": True, "default": ""}),
            "tokenizer": ("TOKENIZER",),
            "res": ("INT", {"default": 16, "min": 0, "max": 50, "step": 1, "display": "slider"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "show_crossattn"
    CATEGORY = "VisualAttn"

    def show_crossattn(self, controller, prompt, tokenizer, res, from_where=("up", "down"), select: int = 0):
        attention_store = controller
        tokens = tokenizer.encode(prompt[select])
        decoder = tokenizer.decode
        attention_maps = aggregate_attention(attention_store, prompt, res, from_where, True, select)

        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        images = np.stack(images, axis=0)
        return (images,)


class ShowSelfAttn:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "controller": ("CONTROLLER",),
            "prompt": ("STRING", {"forceInput": True, "default": ""}),
            "tokenizer": ("TOKENIZER",),
            "res": ("INT", {"default": 16, "min": 0, "max": 50, "step": 1, "display": "slider"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "show_selfattn"
    CATEGORY = "VisualAttn"

    def show_selfattn(self, controller, prompt, tokenizer, res, from_where=("up", "down"), select: int = 0, max_com=10):
        attention_store = controller
        tokens = tokenizer.encode(prompt[select])
        decoder = tokenizer.decode
        attention_maps = aggregate_attention(attention_store, prompt, res, from_where, False, select).numpy().reshape(
            (res ** 2, res ** 2))
        u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))

        images = []
        for i in range(len(tokens)):
            image = vh[i].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        images = np.concatenate(images, axis=1)
        return (images,)


NODE_CLASS_MAPPINGS = {
    "HFModelLoader": HFModelLoader,
    "ShowImages": ShowImages,
    "Text2ImageInference": Text2ImageInference,
    "DecodeLatent": DecodeLatent,
    "ShowCrossAttn": ShowCrossAttn,
    "ShowSelfAttn": ShowSelfAttn
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HFModelLoader": "HF ModelLoader",
    "ShowImages": "Show Images",
    "Text2ImageInference": "Text2Image Inference",
    "DecodeLatent": "Decode Latent",
    "ShowCrossAttn": "Show CrossAttn Map",
    "ShowSelfAttn": "Show SelfAttn Map"
}
