# -*- coding: utf-8 -*-
# HiDream Sampler Node for ComfyUI
# Version: 2024-07-29c (NF4/FP8/BNB Support, Final Call Arg Fix)
#
# Required Dependencies:
# - transformers, diffusers, torch, numpy, Pillow
# - For NF4 models: optimum, accelerate, auto-gptq (`pip install optimum accelerate auto-gptq`)
# - For non-NF4/FP8 models (4-bit): bitsandbytes (`pip install bitsandbytes`)
# - Ensure hi_diffusers library is locally available or hdi1 package is installed.
import torch
import numpy as np
from PIL import Image
import comfy.model_management # Ensure this is imported
import comfy.utils
import gc
import os # For checking paths if needed
# --- Optional Dependency Handling ---
try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False
    print("Warning: accelerate not installed. device_map='auto' for GPTQ models will not be available.")
try:
    import auto_gptq
    autogptq_available = True
except ImportError:
    autogptq_available = False
    # Note: Optimum might still load GPTQ without auto-gptq if using ExLlama kernels,
    # but it's often required. Add a warning if NF4 models are selected later.
try:
    import optimum
    optimum_available = True
except ImportError:
    optimum_available = False
    # Add a warning if NF4 models are selected later.
try:
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    bnb_available = True
except ImportError:
    # This case was handled before, just confirm variable state
    bnb_available = False
    TransformersBitsAndBytesConfig = None
    DiffusersBitsAndBytesConfig = None
    print("Warning: bitsandbytes not installed. 4-bit BNB quantization will not be available.")
# --- Core Imports ---
from transformers import LlamaForCausalLM, AutoTokenizer # Use AutoTokenizer
# --- HiDream Specific Imports ---
# Attempt local import first, then fallback (which might fail)
try:
    # Assuming hi_diffusers is cloned into this custom_node's directory
    from .hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
    from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
    from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
    hidream_classes_loaded = True
except ImportError as e:
    print("--------------------------------------------------------------------")
    print(f"ComfyUI-HiDream-Sampler: Could not import local hi_diffusers ({e}).")
    print("Please ensure hi_diffusers library is inside ComfyUI-HiDream-Sampler,")
    print("or hdi1 package is installed in the ComfyUI environment.")
    print("Node may fail to load models.")
    print("--------------------------------------------------------------------")
    # Define placeholders so the script doesn't crash immediately
    HiDreamImageTransformer2DModel = None
    HiDreamImagePipeline = None
    FlowUniPCMultistepScheduler = None
    FlashFlowMatchEulerDiscreteScheduler = None
    hidream_classes_loaded = False
# --- Model Paths ---
ORIGINAL_MODEL_PREFIX = "HiDream-ai"
NF4_MODEL_PREFIX = "azaneko"
FP8_MODEL_PATH = "shuttleai/HiDream-I1-Full-FP8" # Specific path for FP8 version
ORIGINAL_LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" # For original/FP8
NF4_LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" # For NF4
# --- Model Configurations ---
# Added flags for dependency checks
MODEL_CONFIGS = {
    # --- NF4 Models ---
    "full-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler", # Use string names for dynamic import
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "dev-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "fast-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
     # --- FP8 Model ---
    "full-fp8": {
        "path": FP8_MODEL_PATH,
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False, "is_fp8": True, "requires_bnb": True, "requires_gptq_deps": False # LLM still uses BNB
    },
    # --- Original/BNB Models ---
     "full": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "dev": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "fast": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    }
}
# --- Filter models based on available dependencies ---
# (Keep filtering logic the same)
original_model_count = len(MODEL_CONFIGS)
if not bnb_available: MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_bnb", False)}
if not optimum_available or not autogptq_available: MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_gptq_deps", False)}
if not hidream_classes_loaded: MODEL_CONFIGS = {}
filtered_model_count = len(MODEL_CONFIGS)
if filtered_model_count == 0: print("*"*70 + "\nCRITICAL ERROR: No HiDream models available...\n" + "*"*70)
elif filtered_model_count < original_model_count: print("*"*70 + "\nWarning: Some HiDream models disabled...\n" + "*"*70)
# Define BitsAndBytes configs (if available)
# (Keep definitions the same)
bnb_llm_config = None; bnb_transformer_4bit_config = None
if bnb_available: bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True); bnb_transformer_4bit_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)
model_dtype = torch.bfloat16
# Get available scheduler classes
# (Keep definitions the same)
available_schedulers = {};
if hidream_classes_loaded: available_schedulers = {"FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler, "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler}
# --- Helper: Get Scheduler Instance ---
# (Keep function the same)
def get_scheduler_instance(scheduler_name, shift_value):
    if not available_schedulers: raise RuntimeError("No schedulers available...")
    scheduler_class = available_schedulers.get(scheduler_name)
    if scheduler_class is None: raise ValueError(f"Scheduler class '{scheduler_name}' not found...")
    return scheduler_class(num_train_timesteps=1000, shift=shift_value, use_dynamic_shifting=False)
# --- Loading Function (Handles NF4, FP8, and default BNB) ---
# (Keep function the same as the last version - it correctly sets up the pipe call now)
def load_models(model_type):
    if not hidream_classes_loaded: raise ImportError("Cannot load models: HiDream classes failed to import.")
    if model_type not in MODEL_CONFIGS: raise ValueError(f"Unknown or incompatible model_type: {model_type}")
    config = MODEL_CONFIGS[model_type]
    model_path = config["path"]; is_nf4 = config.get("is_nf4", False); is_fp8 = config.get("is_fp8", False)
    scheduler_name = config["scheduler_class"]; shift = config["shift"]
    requires_bnb = config.get("requires_bnb", False); requires_gptq_deps = config.get("requires_gptq_deps", False)
    if requires_bnb and not bnb_available: raise ImportError(f"Model '{model_type}' requires BitsAndBytes...")
    if requires_gptq_deps and (not optimum_available or not autogptq_available): raise ImportError(f"Model '{model_type}' requires Optimum & AutoGPTQ...")
    print(f"--- Loading Model Type: {model_type} ---"); print(f"Model Path: {model_path}")
    print(f"NF4: {is_nf4}, FP8: {is_fp8}, Requires BNB: {requires_bnb}, Requires GPTQ deps: {requires_gptq_deps}")
    start_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0; print(f"(Start VRAM: {start_mem:.2f} MB)")
    # --- 1. Load LLM (Conditional) ---
    text_encoder_load_kwargs = {"output_hidden_states": True, "low_cpu_mem_usage": True, "torch_dtype": model_dtype,}
    if is_nf4:
        llama_model_name = NF4_LLAMA_MODEL_NAME; print(f"\n[1a] Preparing LLM (GPTQ): {llama_model_name}")
        if accelerate_available: text_encoder_load_kwargs["device_map"] = "auto"; print("     Using device_map='auto'.")
        else: print("     accelerate not found, attempting manual placement.")
    else:
        llama_model_name = ORIGINAL_LLAMA_MODEL_NAME; print(f"\n[1a] Preparing LLM (4-bit BNB): {llama_model_name}")
        if bnb_llm_config: text_encoder_load_kwargs["quantization_config"] = bnb_llm_config; print("     Using 4-bit BNB.")
        else: raise ImportError("BNB config required for standard LLM.")
        text_encoder_load_kwargs["attn_implementation"] = "flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"
    print(f"[1b] Loading Tokenizer: {llama_model_name}..."); tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False); print("     Tokenizer loaded.")
    print(f"[1c] Loading Text Encoder: {llama_model_name}... (May download files)"); text_encoder = LlamaForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)
    if "device_map" not in text_encoder_load_kwargs: print("     Moving text encoder to CUDA..."); text_encoder.to("cuda")
    step1_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0; print(f"✅ Text encoder loaded! (VRAM: {step1_mem:.2f} MB)")
    # --- 2. Load Transformer (Conditional) ---
    print(f"\n[2] Preparing Transformer from: {model_path}"); transformer_load_kwargs = {"subfolder": "transformer", "torch_dtype": model_dtype, "low_cpu_mem_usage": True}
    if is_nf4: print("     Type: NF4")
    elif is_fp8: print("     Type: FP8")
    else: # Default BNB case
        print("     Type: Standard (Applying 4-bit BNB quantization)")
        if bnb_transformer_4bit_config:
            transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config
        else:
            raise ImportError("BNB config required for transformer but unavailable.")
    print("     Loading Transformer... (May download files)"); transformer = HiDreamImageTransformer2DModel.from_pretrained(model_path, **transformer_load_kwargs)
    print("     Moving Transformer to CUDA..."); transformer.to("cuda")
    step2_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0; print(f"✅ Transformer loaded! (VRAM: {step2_mem:.2f} MB)")
    # --- 3. Load Scheduler ---
    print(f"\n[3] Preparing Scheduler: {scheduler_name}"); scheduler = get_scheduler_instance(scheduler_name, shift); print(f"     Using Scheduler: {scheduler_name}")
    # --- 4. Load Pipeline ---
    print(f"\n[4] Loading Pipeline from: {model_path}"); print("     Passing pre-loaded components...")
    pipe = HiDreamImagePipeline.from_pretrained(model_path, scheduler=scheduler, tokenizer_4=tokenizer, text_encoder_4=text_encoder, transformer=None, torch_dtype=model_dtype, low_cpu_mem_usage=True); print("     Pipeline structure loaded.")
    # --- 5. Final Setup ---
    print("\n[5] Finalizing Pipeline..."); print("     Assigning transformer..."); pipe.transformer = transformer
    print("     Moving pipeline object to CUDA (final check)...");
    try: pipe.to("cuda")
    except Exception as e: print(f"     Warning: Could not move pipeline object to CUDA: {e}.")
    if is_nf4:
        print("     Attempting CPU offload for NF4...");
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            try: pipe.enable_sequential_cpu_offload(); print("     ✅ CPU offload enabled.")
            except Exception as e: print(f"     ⚠️ Failed CPU offload: {e}")
        else: print("     ⚠️ enable_sequential_cpu_offload() not found.")
    final_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0; print(f"✅ Pipeline ready! (VRAM: {final_mem:.2f} MB)")
    return pipe, config
# --- Resolution Parsing & Tensor Conversion ---
RESOLUTION_OPTIONS = [ # (Keep list the same)
    "1024 × 1024 (Square)","768 × 1360 (Portrait)","1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)","1168 × 880 (Landscape)","1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]
def parse_resolution(resolution_str): # (Keep function the same)
    try: # ... (same parsing logic) ...
        res_part = resolution_str.split(" (")[0]; parts = res_part.replace('x', '×').split("×")
        if len(parts) != 2: raise ValueError("Format error."); w_str, h_str = [p.strip() for p in parts]
        width = int(w_str); height = int(h_str); return height, width
    except Exception as e: print(f"Error parsing resolution '{resolution_str}': {e}. Falling back."); return 1024, 1024
def pil2tensor(image: Image.Image): # (Keep function the same)
    if image is None: return None; return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
# --- ComfyUI Node Definition ---
class HiDreamSampler:
    _model_cache = {}
    @classmethod
    def INPUT_TYPES(s): # (Keep method the same)
        available_model_types = list(MODEL_CONFIGS.keys())
        if not available_model_types: return {"required": {"error": ("STRING", {"default": "No models available...", "multiline": True})}}
        default_model = "fast-nf4" if "fast-nf4" in available_model_types else "fast" if "fast" in available_model_types else available_model_types[0]
        return {"required": {"model_type": (available_model_types, {"default": default_model}),"prompt": ("STRING", {"multiline": True, "default": "..."}),"resolution": (RESOLUTION_OPTIONS, {"default": "1024 × 1024 (Square)"}),"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}), "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}),}}
    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("image",); FUNCTION = "generate"; CATEGORY = "HiDream"
    def generate(self, model_type, prompt, resolution, seed, override_steps, override_cfg, **kwargs):
        if not MODEL_CONFIGS or model_type == "error": print("HiDream Error: No models loaded."); return (torch.zeros((1, 512, 512, 3)),)
        pipe = None; config = None
        # --- Model Loading / Caching ---
        if model_type in self._model_cache: # (Keep cache checking logic the same)
            print(f"Checking cache for {model_type}..."); pipe, config = self._model_cache[model_type]; valid_cache = True
            if pipe is None or config is None or not hasattr(pipe, 'transformer') or pipe.transformer is None: valid_cache = False; print("Invalid cache, reloading..."); del self._model_cache[model_type]; pipe, config = None, None
            if valid_cache: print("Using cached model.")
        if pipe is None: # (Keep cache clearing logic the same)
            if self._model_cache:
                print(f"Clearing ALL cache before loading {model_type}...")
                keys_to_del = list(self._model_cache.keys())
                for key in keys_to_del:
                    print(f"  Removing '{key}'...")
                    try:
                        pipe_to_del, _= self._model_cache.pop(key)
                        del pipe_to_del
                    except Exception:
                        pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Cache cleared.")
            print(f"Loading model for {model_type}...")
            try:
                pipe, config = load_models(model_type)
                self._model_cache[model_type] = (pipe, config)
                print(f"Model {model_type} loaded & cached!")
            except Exception as e:
                print(f"!!! ERROR loading {model_type}: {e}")
                import traceback
                traceback.print_exc()
                return (torch.zeros((1, 512, 512, 3)),)
        if pipe is None or config is None: print("CRITICAL ERROR: Load failed."); return (torch.zeros((1, 512, 512, 3)),)
        # --- Generation Setup ---
        # (Keep setup logic the same)
        is_nf4_current = config.get("is_nf4", False); height, width = parse_resolution(resolution)
        num_inference_steps = override_steps if override_steps >= 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]
        pbar = comfy.utils.ProgressBar(num_inference_steps) # Keep pbar for final update
        try: inference_device = comfy.model_management.get_torch_device()
        except Exception: inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Creating Generator on: {inference_device}"); generator = torch.Generator(device=inference_device).manual_seed(seed)
        print(f"\n--- Starting Generation ---"); print(f"Model: {model_type}, Res: {height}x{width}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
        # --- Run Inference ---
        output_images = None
        try:
            if not is_nf4_current: print(f"Ensuring pipe on: {inference_device} (Offload NOT enabled)"); pipe.to(inference_device)
            else: print(f"Skipping pipe.to({inference_device}) (CPU offload enabled).")
            print("Executing pipeline inference...")
            with torch.inference_mode():
                # *** Final pipe() call matching reference script ***
                output_images = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                ).images
            print("Pipeline inference finished.")
        except Exception as e: print(f"!!! ERROR during execution: {e}"); import traceback; traceback.print_exc(); return (torch.zeros((1, height, width, 3)),)
        finally: pbar.update_absolute(num_inference_steps) # Update pbar regardless
        print("--- Generation Complete ---")
        if not output_images: print("ERROR: No images returned."); return (torch.zeros((1, height, width, 3)),)
        output_tensor = pil2tensor(output_images[0]); return (output_tensor,)
# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {"HiDreamSampler": HiDreamSampler}; NODE_DISPLAY_NAME_MAPPINGS = {"HiDreamSampler": "HiDream Sampler (NF4/FP8/BNB)"}
print("-" * 50 + "\nHiDream Sampler Node Initialized\nAvailable Models: " + str(list(MODEL_CONFIGS.keys())) + "\n" + "-" * 50) # Compact print
