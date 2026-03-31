import subprocess
import sys
import re
import os
import json
import requests
import time
import random
import base64
import io
import gc
import torch
import numpy as np
from PIL import Image
import comfy.model_management as model_management
from urllib.parse import urlparse
class AnyType(str):
    """A special class that is always equal in not-equal comparisons. Allows accepting any input type."""
    def __ne__(self, __value: object) -> bool:
        return False
any_type = AnyType("*")
def tensor_to_base64_jpeg(image_tensor):
    """Converts a ComfyUI image tensor (B, H, W, C) to a base64 JPEG string."""
    img = image_tensor[0].cpu().numpy()
    img = (img * 255).astype(np.uint8)
    pil_image = Image.fromarray(img)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
def get_auth_headers():
    token = (os.environ.get("LMSTUDIO_API_KEY") or
             os.environ.get("LM_API_TOKEN") or
             os.environ.get("LM_API_KEY"))
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}
def send_to_lm_studio(client_config, messages, max_tokens, temp, top_p, seed, json_mode=False):
    """Abstracted API caller with proper error raising."""
    url = client_config.get("url", "http://127.0.0.1:1234/v1/chat/completions")
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": top_p,
        "seed": seed
    }
    if client_config.get("model_identifier"):
        payload["model"] = client_config["model_identifier"]
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, headers=get_auth_headers(), timeout=600)
        if response.status_code == 200:
            result_json = response.json()
            content = result_json["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
            elapsed_time = time.time() - start_time
            print(f"\n[LM Studio Response time: {elapsed_time:.2f} seconds]\n")
            return content if isinstance(content, str) else json.dumps(result_json, indent=2)
        raise ValueError(f"LM Studio API Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"LM Studio Connection Failed: {str(e)}")
class EbuLMStudioLoader:
    """
    Loads a model into LM Studio VRAM via CLI and outputs an LM_CLIENT config.
    """
    ERROR_NO_MODEL_FOUND = "no model by that name found"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_search_string": ("STRING", {"default": "llama"}),
                "url": ("STRING", { "multiline": False, "default": "http://127.0.0.1:1234/v1/chat/completions" }),
                "context_length": ("INT", {"default": 4096, "min": 1000, "max": 200000}),
                "unload_image_models_first": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("LM_CLIENT", "STRING",)
    RETURN_NAMES = ("lm_client", "model_loaded",)
    FUNCTION = "load_model"
    CATEGORY = "LMStudio"
    def run_command(self, command, timeout=180):
        try:
            process = subprocess.run(
                [str(c) for c in command], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout, shell=False
            )
            if process.returncode != 0:
                print(f"[lms-node] Command failed: {command}", file=sys.stderr)
                return None
            return process
        except Exception as e:
            print(f"[lms-node] Unexpected error running {command}: {e}", file=sys.stderr)
            return None
    def _list_models_json(self):
        for cmd in (['lms', 'ls', '--llm', '--json'], ['lms', 'ls', '--json'], ['lms', 'models', 'ls', '--json']):
            p = self.run_command(cmd)
            if not (p and p.stdout and p.stdout.strip().startswith(('[', '{'))): continue
            try:
                data = json.loads(p.stdout)
                items = data.get('models') or data.get('data') or data.get('items') or data if isinstance(data, dict) else data
                if isinstance(items, list): return [m for m in items if isinstance(m, dict)]
            except Exception: pass
        return []
    def _ps_models(self):
        p = self.run_command(['lms', 'ps', '--json'])
        if p and p.stdout and p.stdout.strip().startswith(('[', '{')):
            try:
                data = json.loads(p.stdout)
                items = data.get('models') or data.get('data') or data if isinstance(data, dict) else data
                if isinstance(items, list):
                    return [{"identifier": m.get("identifier") or m.get("id") or m.get("modelKey"), "path": m.get("path")} for m in items if isinstance(m, dict)]
            except Exception: pass
        return []
    def load_model(self, model_search_string, url, context_length, unload_image_models_first):
        models = self._list_models_json()
        tokens = [t for t in re.split(r'\s+', (model_search_string or '').strip().lower()) if t]
        candidates = []
        for m in models:
            key = str(m.get('modelKey') or m.get('identifier') or m.get('name') or "")
            hay = " ".join([key, str(m.get('displayName', "")), str(m.get('path', ""))]).lower()
            if all(t in hay for t in tokens):
                candidates.append(key)
        if not candidates:
            raise ValueError(f"No models found matching '{model_search_string}'.")
        model_key_to_load = candidates[0]
        desired_identifier = re.sub(r'[^A-Za-z0-9._\-]+', '-', model_key_to_load).strip('-')[:64]
        for r in self._ps_models():
            if r.get("identifier") in (model_key_to_load, desired_identifier):
                print(f"Model already loaded: {model_key_to_load}")
                return ({"url": url, "model_identifier": desired_identifier}, desired_identifier)
        if unload_image_models_first:
            print("Unloading ComfyUI image models to save memory...")
            model_management.unload_all_models()
            model_management.soft_empty_cache(True)
            gc.collect()
            torch.cuda.empty_cache()
        print(f"Loading model: {model_key_to_load}")
        load_cmd = ['lms', 'load', model_key_to_load, '-y', '--identifier', desired_identifier, '--context-length', str(int(context_length)), '--gpu', 'max']
        if not self.run_command(load_cmd, timeout=600):
            raise ValueError(f"Failed to load model: {model_key_to_load}")
        return ({"url": url, "model_identifier": desired_identifier}, desired_identifier)
class EbuLMStudioChat:
    """
    Primary API request node. Supports Vision, JSON mode, and Chat History.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lm_client": ("LM_CLIENT",),
                "system_message": ("STRING", { "multiline": True, "default": "You are a helpful assistant." }),
                "user_prompt": ("STRING", { "multiline": True, "default": "" }),
                "max_tokens": ("INT", {"default": 500, "min": 10, "max": 100000}),
                "temp": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
                "json_mode": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "chat_history": ("CHAT_HISTORY",),
            }
        }
    RETURN_TYPES = ("STRING", "CHAT_HISTORY")
    RETURN_NAMES = ("generated_text", "chat_history")
    FUNCTION = "generate_chat"
    CATEGORY = "LMStudio"
    def sanitize_utf8(self, text):
        from unicodedata import normalize
        return normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    def generate_chat(self, lm_client, system_message, user_prompt, max_tokens, temp, top_p, json_mode, seed, image=None, chat_history=None):
        messages = []
        if chat_history is not None:
            messages.extend(chat_history)
        else:
            if system_message.strip():
                messages.append({"role": "system", "content": system_message})
        user_content = user_prompt
        if image is not None:
            base64_image = tensor_to_base64_jpeg(image)
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        messages.append({"role": "user", "content": user_content})
        response_text = send_to_lm_studio(lm_client, messages, max_tokens, temp, top_p, seed, json_mode)
        response_text = self.sanitize_utf8(response_text)
        new_history = list(messages)
        new_history.append({"role": "assistant", "content": response_text})
        return (response_text, new_history)
class EbuLMStudioBrainstormer:
    """
    Generates structured lists of ideas using the shared LM_CLIENT object.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lm_client": ("LM_CLIENT",),
                "topic": ("STRING", {"default": "Fantasy Setting Magic Spells"}),
                "for_each_idea": ("STRING", {"default": "A magic spell that a wizard or sorceress might cast.", "multiline": True}),
                "raw_list_size": ("INT", {"default": 20, "min": 1}),
                "return_list_size": ("INT", {"default": 10, "min": 0}),
                "ignore_the_first": ("INT", {"default": 0, "min": 0}),
                "max_tokens": ("INT", {"default": 500, "min": 10, "max": 100000}),
                "temp": ("FLOAT", {"default": 0.70, "min": 0.00, "max": 3.00, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "full_list")
    FUNCTION = "brainstorm"
    CATEGORY = "LMStudio"
    def brainstorm(self, lm_client, topic, for_each_idea, raw_list_size, return_list_size, ignore_the_first, max_tokens, temp, seed):
        system_message = "You are an expert brainstorming assistant. Respond with the numbered list only—no titles, no commentary. Each suggestion on one line."
        prompt = (
            f"Generate exactly {raw_list_size} ideas on the topic '{topic}'.\n\n"
            f"For each idea, provide exactly this information and no more: {for_each_idea}.\n"
            f"Return exactly {raw_list_size} lines. Each line must begin with '1. ', '2. ', etc."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        full_list = send_to_lm_studio(lm_client, messages, max_tokens, temp, 0.95, seed, json_mode=False).strip()
        lines = [line for line in full_list.splitlines() if line.strip()]
        if ignore_the_first > 0:
            lines = lines[ignore_the_first:]
        if return_list_size == 0: result = ""
        elif return_list_size == 1:
            result = re.sub(r'^\s*\d+[\.|\)]\s*', '', random.choice(lines)).strip()
        else:
            sampled = random.sample(lines, min(return_list_size, len(lines)))
            stripped = [re.sub(r'^\s*\d+[\.|\)]\s*', '', l).strip() for l in sampled]
            result = "\n".join([f"{i+1}. {stripped[i]}" for i in range(len(stripped))])
        return (result, full_list)
class EbuLMStudioUnload:
    """
    Universal Unload Node. Takes any input simply to act as an execution trigger.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_trigger": (any_type,),
            },
        }
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "unload_all"
    CATEGORY = "LMStudio"
    def unload_all(self, any_trigger):
        try:
            print("Calling lms unload --all")
            subprocess.run(['lms', 'unload', '--all'], capture_output=True, text=True, check=True)
            return (any_trigger,)
        except Exception as e:
            print(f"[lms-node] unload_all failed: {e}", file=sys.stderr)
            return (any_trigger,)
NODE_CLASS_MAPPINGS = {
    "EbuLMStudioLoader": EbuLMStudioLoader,
    "EbuLMStudioChat": EbuLMStudioChat,
    "EbuLMStudioBrainstormer": EbuLMStudioBrainstormer,
    "EbuLMStudioUnload": EbuLMStudioUnload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EbuLMStudioLoader": "EBU LMStudio Loader",
    "EbuLMStudioChat": "EBU LMStudio Chat / Vision",
    "EbuLMStudioBrainstormer": "EBU LMStudio Brainstormer",
    "EbuLMStudioUnload": "EBU LMStudio Universal Unload",
}