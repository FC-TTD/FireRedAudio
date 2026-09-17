"""Importable hooks for the isolated GPU process."""
def load_model():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Managed FireRed requires its assigned CUDA GPU")
    from .native import NativeBackend
    return NativeBackend()

def completion(model):
    import torch
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()

def release(model):
    model.close()

def cleanup():
    import gc
    gc.collect()
