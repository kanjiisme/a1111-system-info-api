import sys
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modules.api.models import *
from modules.api import api
import psutil
import torch
from modules import paths, script_callbacks, sd_hijack, sd_models, sd_samplers, shared, extensions, devices
import platform
import subprocess
import accelerate
import os
import transformers
import uuid
try:
    import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
except:
    pass

def get_loras():
    loras = []
    try:
        sys.path.append(extensions.extensions_builtin_dir)
        from Lora import lora # pylint: disable=E0401
        loras = sorted([l for l in lora.available_loras.keys()])
    except:
        pass
    return loras


_server_id = uuid.uuid1().hex

print("You worker id: ",  _server_id)

def get_platform():
    try:
        if platform.system() == 'Windows':
            release = platform.platform(aliased = True, terse = False)
        else:
            release = platform.release()
        return {
            # 'host': platform.node(),
            'arch': platform.machine(),
            'system': platform.system(),
            'release': release,
            # 'platform': platform.platform(aliased = True, terse = False),
            # 'version': platform.version(),
            'python': platform.python_version(),
        }
    except Exception as e:
        return { 'error': e }


def get_torch():
    try:
        ver = torch.__long_version__
    except:
        ver = torch.__version__
    return f"{ver} {shared.cmd_opts.precision} {' nohalf' if shared.cmd_opts.no_half else ' half'}"


def get_optimizations():
    ram = []
    if shared.cmd_opts.medvram:
        ram.append('medvram')
    if shared.cmd_opts.lowvram:
        ram.append('lowvram')
    if shared.cmd_opts.lowram:
        ram.append('lowram')
    if len(ram) == 0:
        ram.append('none')
    return ram


def get_libs():
    try:
        import xformers # pylint: disable=import-outside-toplevel, import-error
        xversion = xformers.__version__
    except:
        xversion = 'unavailable'
    return {
        'xformers': xversion,
        'accelerate': accelerate.__version__,
        'transformers': transformers.__version__,
    }

def get_gpu():
    if not torch.cuda.is_available():
        try:
            return {
                'device': f'{torch.xpu.get_device_name(torch.xpu.current_device())} ({str(torch.xpu.device_count())})',
                'ipex': str(ipex.__version__),
            }
        except:
            return {}
    else:
        try:
            if torch.version.cuda:
                return {
                    'device': f'{torch.cuda.get_device_name(torch.cuda.current_device())} ({str(torch.cuda.device_count())}) ({torch.cuda.get_arch_list()[-1]}) {str(torch.cuda.get_device_capability(shared.device))}',
                    'cuda': torch.version.cuda,
                    'cudnn': torch.backends.cudnn.version(),
                }
            elif torch.version.hip:
                return {
                    'device': f'{torch.cuda.get_device_name(torch.cuda.current_device())} ({str(torch.cuda.device_count())})',
                    'hip': torch.version.hip,
                }
            else:
                return {
                    'device': 'unknown'
                }
        except Exception as e:
            return { 'error': e }
        
def get_crossattention():
    try:
        ca = sd_hijack.model_hijack.optimization_method
        if ca is None:
            return 'none'
        else: return ca
    except:
        return 'unknown'
        
def diffusion_worker_api(_ : gr.Blocks, app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Set the appropriate origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.get("/worker/sysinfo")
    async def get_info():
        s = torch.cuda.mem_get_info()
        return {
            "id" : _server_id,
            "optimizations" : get_optimizations(),
            "libs" : get_libs(),
            "get_platform" : get_platform(),
            "hardware" : {
                'cpu': {
                    'cpu': platform.processor(),
                    'core' : psutil.cpu_count(False),
                    'logical' : psutil.cpu_count(),
                    'speed' : psutil.cpu_freq()[2], 
                    'used': psutil.cpu_freq()[0]
                },
                "ram" : {
                    "memory" : psutil.virtual_memory().total,
                    "used" : psutil.virtual_memory().used
                },
                "gpu" : {
                    "name" : get_gpu(),
                    "memory" : s[1],
                    "used" : s[1] - s[0]
                }
            },
            'crossattention': get_crossattention(),
        }
    
    @app.get("/worker/loras")
    async def get_loras_list():
        return get_loras()
    
try:
    script_callbacks.on_app_started(diffusion_worker_api)
except Exception as e:
    print(e)