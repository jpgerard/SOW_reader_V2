"""
PyTorch utilities and initialization
"""

def initialize_torch():
    """Initialize PyTorch with settings for inference"""
    try:
        import torch
        torch.set_grad_enabled(False)  # Disable gradients since we're only doing inference
        if hasattr(torch, 'jit'):
            torch.jit.disable()
        return torch
    except ImportError:
        return None
    except Exception as e:
        print(f"PyTorch initialization warning: {str(e)}")
        return None
