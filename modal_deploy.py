"""
Modal deployment file for IDM-VTON API.
Deploys the virtual try-on application with A100 GPU support.
Models are cached in a Modal Volume to avoid re-downloading.
"""

from pathlib import Path
import modal

def ignore_local_file(path: Path) -> bool:
    # Skip uploading local checkpoint folders so build-time downloads remain
    return path.parts and path.parts[0] == "ckpt"

# Define the Modal app
app = modal.App("idm-vton")

# Create a volume for caching models (persists across deployments)
model_volume = modal.Volume.from_name("idm-vton-models", create_if_missing=True)

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "curl",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "build-essential",
    )
    .run_commands(
        # Install PyTorch with CUDA 11.8 support (matching environment.yaml)
        "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "accelerate==0.25.0",
        "torchmetrics==1.2.1",
        "tqdm==4.66.1",
        "transformers==4.36.2",
        "diffusers==0.25.0",
        "huggingface-hub==0.19.4",
        "einops==0.7.0",
        "bitsandbytes==0.39.0",
        "scipy==1.11.1",
        "opencv-python",
        "fastapi==0.110.0",
        "python-multipart",
        "ultralytics==8.3.25",
        "fvcore",
        "cloudpickle",
        "omegaconf",
        "pycocotools",
        "basicsr",
        "av",
        "onnxruntime==1.16.2",
        "Pillow",
        "numpy",
        "packaging",
    )
    .run_commands(
        # Install detectron2 - build from source for CUDA 11.8 compatibility
        # Note: This may take several minutes during image build
        "pip install 'git+https://github.com/facebookresearch/detectron2.git'",
    )
    .run_commands(
        # Install additional dependencies that might be needed
        "pip install pyyaml",
    )
    .run_commands(
        # Clone the repository
        "git clone https://github.com/yisol/IDM-VTON.git /root/IDM-VTON",
        "cd /root/IDM-VTON && git checkout main",
    )
    .run_commands(
        # Create checkpoint directories
        "mkdir -p /root/IDM-VTON/ckpt/densepose",
        "mkdir -p /root/IDM-VTON/ckpt/humanparsing",
        "mkdir -p /root/IDM-VTON/ckpt/openpose/ckpts",
    )
    .run_commands(
        # Download DensePose checkpoint
        "cd /root/IDM-VTON/ckpt/densepose && "
        "wget -q https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl -O model_final_162be9.pkl",
    )
    .run_commands(
        # Download human parsing checkpoints
        "cd /root/IDM-VTON/ckpt/humanparsing && "
        "wget -q https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx -O parsing_atr.onnx && "
        "wget -q https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx -O parsing_lip.onnx",
    )
    .run_commands(
        # Download OpenPose checkpoint
        "cd /root/IDM-VTON/ckpt/openpose/ckpts && "
        "wget -q https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth -O body_pose_model.pth",
    )
    .add_local_dir(".", remote_path="/root/IDM-VTON", ignore=ignore_local_file)
)

@app.cls(
    image=image,
    gpu="L4",
    timeout=3600,
    scaledown_window=300,
    volumes={"/models": model_volume},
    container_idle_timeout=300,
)
class TryOnModel:
    """Model class that loads models once on container startup."""
    
    @modal.enter()
    def load_models(self):
        """Load all models once when container starts."""
        import sys
        import os
        import torch
        from PIL import Image
        from typing import List
        from torchvision import transforms
        from torchvision.transforms.functional import to_pil_image
        import numpy as np
        from ultralytics import YOLO

        # Set HuggingFace cache to volume for persistence
        hf_cache_dir = "/models/huggingface"
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir

        # Reuse Modal volume for Ultralytics downloads as well
        yolo_cache_dir = "/models/ultralytics"
        os.makedirs(yolo_cache_dir, exist_ok=True)
        os.environ.setdefault("ULTRALYTICS_CACHE_DIR", yolo_cache_dir)
        os.environ.setdefault("ULTRALYTICS_ASSETS_DIR", yolo_cache_dir)

        # Add project root (and gradio_demo) to sys.path so internal packages resolve
        sys.path.insert(0, "/root/IDM-VTON")
        sys.path.insert(0, "/root/IDM-VTON/gradio_demo")
        
        # Make densepose importable as a top-level module for apply_net.py
        import importlib
        densepose_module = importlib.import_module("densepose")
        sys.modules["densepose"] = densepose_module

        # Set device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
        from src.unet_hacked_tryon import UNet2DConditionModel
        from transformers import (
            CLIPImageProcessor,
            CLIPVisionModelWithProjection,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            AutoTokenizer,
        )
        from diffusers import DDPMScheduler, AutoencoderKL
        from gradio_demo.utils_mask import get_mask_location
        import gradio_demo.apply_net as apply_net
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose
        from detectron2.data.detection_utils import (
            convert_PIL_to_numpy,
            _apply_exif_orientation,
        )

        # Change to project directory
        os.chdir("/root/IDM-VTON")

        def pil_to_binary_mask(pil_image, threshold=0):
            np_image = np.array(pil_image)
            grayscale_image = Image.fromarray(np_image).convert("L")
            binary_mask = np.array(grayscale_image) > threshold
            mask = np.zeros(binary_mask.shape, dtype=np.uint8)
            for i in range(binary_mask.shape[0]):
                for j in range(binary_mask.shape[1]):
                    if binary_mask[i, j] == True:
                        mask[i, j] = 1
            mask = (mask * 255).astype(np.uint8)
            output_mask = Image.fromarray(mask)
            return output_mask

        # Load models from HuggingFace (will use cache from volume if available)
        base_path = "yisol/IDM-VTON"
        print("Loading models from HuggingFace (using volume cache if available)...")

        self.unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir,
        )
        self.unet.requires_grad_(False)

        self.tokenizer_one = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
            cache_dir=hf_cache_dir,
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
            cache_dir=hf_cache_dir,
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            base_path, subfolder="scheduler", cache_dir=hf_cache_dir
        )

        self.text_encoder_one = CLIPTextModel.from_pretrained(
            base_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir,
        )

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir,
        )

        self.vae = AutoencoderKL.from_pretrained(
            base_path,
            subfolder="vae",
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir,
        )

        self.UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir,
        )

        # Initialize preprocessing models
        print("Loading preprocessing models...")
        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)

        # Freeze all models
        self.UNet_Encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)

        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Create pipeline
        print("Creating pipeline...")
        self.pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=self.unet,
            vae=self.vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            scheduler=self.noise_scheduler,
            image_encoder=self.image_encoder,
            torch_dtype=torch.float16,
            cache_dir=hf_cache_dir,
        )
        self.pipe.unet_encoder = self.UNet_Encoder

        # Store helper functions and imports
        self.pil_to_binary_mask = pil_to_binary_mask
        self.get_mask_location = get_mask_location
        self.apply_net = apply_net
        self.convert_PIL_to_numpy = convert_PIL_to_numpy
        self._apply_exif_orientation = _apply_exif_orientation
        self.to_pil_image = to_pil_image
        self.List = List

        # Load YOLO person detector once for smart cropping
        print("Loading YOLO person detector for auto-cropping...")
        self.person_detector = YOLO("yolov8n.pt")
        self.person_detector.to(self.device)
        self.detector_confidence = 0.35
        self.max_process_side = 1024
        self.default_size = (768, 1024)

        print("Models loaded successfully! Volume cache will persist across deployments.")

    def _compute_target_size(self, region_size):
        """Compute model resolution (multiples of 8) that preserves region aspect ratio."""
        width, height = region_size
        if width <= 0 or height <= 0:
            return self.default_size

        long_side = max(width, height)
        scale = self.max_process_side / float(long_side)
        scaled_w = width * scale
        scaled_h = height * scale

        target_width = max(64, int(round(scaled_w / 8.0)) * 8)
        target_height = max(64, int(round(scaled_h / 8.0)) * 8)
        return target_width, target_height

    def _auto_crop_with_yolo(self, image):
        """Detect the person with YOLO and return a tight crop plus metadata."""
        import numpy as np

        if not hasattr(self, "person_detector"):
            return None

        np_image = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR for YOLO
        results = self.person_detector.predict(
            source=np_image,
            classes=[0],  # person class
            conf=self.detector_confidence,
            verbose=False,
            device=self.device,
        )

        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
            return None

        confidences = boxes.conf.detach().cpu().numpy()
        best_idx = int(confidences.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].detach().cpu().numpy()

        width, height = image.size
        margin = 0.08
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            return None

        x1 -= box_w * margin
        y1 -= box_h * margin
        x2 += box_w * margin
        y2 += box_h * margin

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        crop_box = (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        )
        cropped = image.crop(crop_box)

        return cropped, {
            "left": crop_box[0],
            "top": crop_box[1],
            "right": crop_box[2],
            "bottom": crop_box[3],
            "crop_size": cropped.size,
            "original": image.copy(),
        }

    def preprocess_human_image(self, dict, is_checked, is_checked_crop):
        """Preprocess human image once - segmentation, pose, mask (reusable for batch)."""
        from PIL import Image
        from torchvision.transforms.functional import to_pil_image
        
        human_img_orig = dict["background"].convert("RGB")
        crop_info = None
        work_img = human_img_orig

        if is_checked_crop:
            yolo_result = self._auto_crop_with_yolo(human_img_orig)
            if yolo_result:
                work_img, crop_info = yolo_result
            else:
                width, height = human_img_orig.size
                target_width = int(min(width, height * (3 / 4)))
                target_height = int(min(height, width * (4 / 3)))
                left = (width - target_width) / 2
                top = (height - target_height) / 2
                right = (width + target_width) / 2
                bottom = (height + target_height) / 2
                work_img = human_img_orig.crop((left, top, right, bottom))
                crop_info = {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "crop_size": work_img.size,
                    "original": human_img_orig.copy(),
                }

            if crop_info and "original" not in crop_info:
                crop_info["original"] = human_img_orig.copy()

        target_size = self._compute_target_size(work_img.size)
        human_img = work_img.resize(target_size)
        if crop_info:
            crop_info["crop_size"] = work_img.size

        if is_checked:
            keypoints = self.openpose_model(human_img.resize((384, 512)))
            model_parse, _ = self.parsing_model(human_img.resize((384, 512)))
            mask, mask_gray = self.get_mask_location(
                "hd", "upper_body", model_parse, keypoints
            )
            mask = mask.resize(target_size)
        else:
            if dict.get("layers") and len(dict["layers"]) > 0:
                mask = self.pil_to_binary_mask(
                    dict["layers"][0].convert("RGB").resize(target_size)
                )
            else:
                # Fallback: create a simple mask if no layers provided
                mask = Image.new("L", target_size, 255)

        mask_gray = (1 - self.tensor_transform(mask)) * self.tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        # DensePose processing (expensive - only do once)
        human_img_arg = self._apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = self.convert_PIL_to_numpy(human_img_arg, format="BGR")

        args = self.apply_net.create_argument_parser().parse_args(
            (
                "show",
                "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
                "./ckpt/densepose/model_final_162be9.pkl",
                "dp_segm",
                "-v",
                "--opts",
                "MODEL.DEVICE",
                "cuda",
            )
        )
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize(target_size)

        return {
            "human_img": human_img,
            "mask": mask,
            "mask_gray": mask_gray,
            "pose_img": pose_img,
            "crop_info": crop_info,
            "target_size": target_size,
        }

    def run_diffusion_only(self, preprocessed_data, garm_img, garment_des, denoise_steps, seed):
        """Run only the diffusion part with pre-processed human data."""
        import torch
        from PIL import Image
        
        self.openpose_model.preprocessor.body_estimation.model.to(self.device)
        self.pipe.to(self.device)
        self.pipe.unet_encoder.to(self.device)

        target_size = preprocessed_data.get("target_size", self.default_size)
        garm_img = garm_img.convert("RGB").resize(target_size)
        human_img = preprocessed_data["human_img"]
        mask = preprocessed_data["mask"]
        pose_img = preprocessed_data["pose_img"]
        crop_info = preprocessed_data["crop_info"]

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                prompt = "model is wearing " + garment_des
                negative_prompt = (
                    "monochrome, lowres, bad anatomy, worst quality, low quality"
                )
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = self.pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt = "a photo of " + garment_des
                    negative_prompt = (
                        "monochrome, lowres, bad anatomy, worst quality, low quality"
                    )
                    if not isinstance(prompt, self.List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, self.List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = self.pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img_tensor = (
                        self.tensor_transform(pose_img).unsqueeze(0).to(self.device, torch.float16)
                    )
                    garm_tensor = (
                        self.tensor_transform(garm_img).unsqueeze(0).to(self.device, torch.float16)
                    )
                    generator = (
                        torch.Generator(self.device).manual_seed(int(seed))
                        if seed is not None and seed >= 0
                        else None
                    )
                    images = self.pipe(
                        prompt_embeds=prompt_embeds.to(self.device, torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(
                            self.device, torch.float16
                        ),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(
                            self.device, torch.float16
                        ),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(
                            self.device, torch.float16
                        ),
                        num_inference_steps=int(denoise_steps),
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img_tensor.to(self.device, torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(self.device, torch.float16),
                        cloth=garm_tensor.to(self.device, torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=int(target_size[1]),
                        width=int(target_size[0]),
                        ip_adapter_image=garm_img,
                        guidance_scale=2.0,
                    )[0]

        if crop_info:
            out_img = images[0].resize(crop_info["crop_size"])
            crop_info["original"].paste(out_img, (int(crop_info["left"]), int(crop_info["top"])))
            return crop_info["original"]
        else:
            return images[0]

    def start_tryon(
        self, dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed
    ):
        """Run the try-on pipeline (full process - for single requests)."""
        preprocessed = self.preprocess_human_image(dict, is_checked, is_checked_crop)
        output_image = self.run_diffusion_only(preprocessed, garm_img, garment_des, denoise_steps, seed)
        return output_image, preprocessed["mask_gray"]


    @modal.asgi_app()
    def api(self):
        """FastAPI app that accepts file uploads and returns actual image files."""
        from fastapi import FastAPI, File, UploadFile, Form, HTTPException
        from fastapi.responses import Response
        from PIL import Image
        from io import BytesIO

        api_app = FastAPI(title="IDM-VTON API", version="1.0.0")

        @api_app.post("/tryon")
        async def run_tryon(
            human_image: UploadFile = File(..., description="Human image file (required)"),
            garment_image: UploadFile = File(..., description="Garment image file (required)"),
            garment_description: str = Form(None, description="Text description of the garment (optional)"),
            auto_mask: bool = Form(None, description="Use auto-generated mask (optional, defaults to True)"),
            auto_crop: bool = Form(None, description="Auto-crop and resize the human image (optional, defaults to False)"),
            denoise_steps: int = Form(None, ge=20, le=40, description="Denoising steps (optional, defaults to 30)"),
            seed: int = Form(None, ge=-1, description="Random seed (optional, defaults to 42)"),
            mask_image: UploadFile = File(None, description="Optional mask image (only used when auto_mask is false)"),
        ):
            try:
                # Apply defaults for optional parameters
                garment_desc = garment_description if garment_description else "garment"
                use_auto_mask = auto_mask if auto_mask is not None else True
                use_auto_crop = auto_crop if auto_crop is not None else False
                steps = denoise_steps if denoise_steps is not None else 30
                use_seed = seed if seed is not None else 42
                
                # Read uploaded images
                human_img_data = await human_image.read()
                garment_img_data = await garment_image.read()
                
                human_img = Image.open(BytesIO(human_img_data)).convert("RGB")
                garment_img = Image.open(BytesIO(garment_img_data)).convert("RGB")

                input_dict = {"background": human_img}
                
                # Handle optional mask image (only if auto_mask is False)
                if not use_auto_mask and mask_image and mask_image.filename:
                    mask_img_data = await mask_image.read()
                    mask_img = Image.open(BytesIO(mask_img_data)).convert("RGB")
                    input_dict["layers"] = [mask_img]

                # Run try-on using the shared model instance
                output_image, _ = self.start_tryon(
                    input_dict,
                    garment_img,
                    garment_desc,
                    use_auto_mask,
                    use_auto_crop,
                    steps,
                    use_seed,
                )

                # Return output image as PNG (no mask)
                output_buffer = BytesIO()
                output_image.save(output_buffer, format="PNG")
                output_buffer.seek(0)
                
                return Response(
                    content=output_buffer.getvalue(),
                    media_type="image/png",
                    headers={"Content-Disposition": "attachment; filename=output.png"}
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

        @api_app.post("/tryon/batch")
        async def run_tryon_batch(
            human_image: UploadFile = File(..., description="Human image file (required)"),
            garment_images: list[UploadFile] = File(..., description="Multiple garment image files (required)"),
            garment_descriptions: str = Form(None, description="Comma-separated descriptions for each garment (optional)"),
            auto_mask: bool = Form(None, description="Use auto-generated mask (optional, defaults to True)"),
            auto_crop: bool = Form(None, description="Auto-crop and resize the human image (optional, defaults to False)"),
            denoise_steps: int = Form(None, ge=20, le=40, description="Denoising steps (optional, defaults to 30)"),
            seed: int = Form(None, ge=-1, description="Random seed (optional, defaults to 42)"),
            mask_image: UploadFile = File(None, description="Optional mask image (only used when auto_mask is false)"),
        ):
            """Process one person image with multiple garment images and return all results.
            OPTIMIZED: Preprocesses person image once, then runs diffusion for each garment."""
            import zipfile
            
            try:
                # Apply defaults for optional parameters
                use_auto_mask = auto_mask if auto_mask is not None else True
                use_auto_crop = auto_crop if auto_crop is not None else False
                steps = denoise_steps if denoise_steps is not None else 30
                use_seed = seed if seed is not None else 42
                
                # Read human image once
                human_img_data = await human_image.read()
                human_img = Image.open(BytesIO(human_img_data)).convert("RGB")
                
                input_dict = {"background": human_img}
                
                # Handle optional mask image (only if auto_mask is False)
                if not use_auto_mask and mask_image and mask_image.filename:
                    mask_img_data = await mask_image.read()
                    mask_img = Image.open(BytesIO(mask_img_data)).convert("RGB")
                    input_dict["layers"] = [mask_img]

                # PREPROCESS HUMAN IMAGE ONCE (expensive operations: segmentation, pose, mask)
                # This saves significant time and processing power for batch requests
                print(f"Preprocessing human image once for {len(garment_images)} garments...")
                preprocessed_data = self.preprocess_human_image(input_dict, use_auto_mask, use_auto_crop)
                print("Human image preprocessing complete. Processing garments...")

                # Parse garment descriptions if provided
                descriptions_list = None
                if garment_descriptions:
                    descriptions_list = [desc.strip() for desc in garment_descriptions.split(",")]
                
                # Process each garment image (only diffusion, no re-preprocessing)
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, garment_file in enumerate(garment_images):
                        try:
                            # Read garment image
                            garment_img_data = await garment_file.read()
                            garment_img = Image.open(BytesIO(garment_img_data)).convert("RGB")
                            
                            # Get description for this garment
                            if descriptions_list and idx < len(descriptions_list):
                                garment_desc = descriptions_list[idx]
                            else:
                                garment_desc = "garment"  # Default if no description provided
                            
                            # Run ONLY diffusion (human image already preprocessed)
                            print(f"Processing garment {idx + 1}/{len(garment_images)}: {garment_desc}")
                            output_image = self.run_diffusion_only(
                                preprocessed_data,
                                garment_img,
                                garment_desc,
                                steps,
                                use_seed,
                            )
                            
                            # Save to zip with descriptive filename
                            output_buffer = BytesIO()
                            output_image.save(output_buffer, format="PNG")
                            filename = f"output_{idx + 1}_{garment_desc.replace(' ', '_')}.png"
                            zip_file.writestr(filename, output_buffer.getvalue())
                            
                        except Exception as e:
                            # If one garment fails, continue with others
                            error_filename = f"error_{idx + 1}.txt"
                            zip_file.writestr(error_filename, f"Error processing garment {idx + 1}: {str(e)}")
                
                zip_buffer.seek(0)
                
                return Response(
                    content=zip_buffer.getvalue(),
                    media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=tryon_batch_results.zip"}
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing batch request: {str(e)}")

        @api_app.get("/health")
        async def health():
            return {"status": "healthy", "models_loaded": True}

        return api_app
