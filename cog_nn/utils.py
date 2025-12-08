
"""
Utility functions.
"""
import numpy as np
import torch


def forward_backward_hook(name, storage_dict):
    """
    Returns a hook function to track:
    - Forward activations
    - Backward gradients
    """
    def forward_hook(module, input, output):
        # Handle tuple/list outputs (some modules return multiple tensors)
        if isinstance(output, torch.Tensor):
            storage_dict[name]['activation'] = output.detach()

            # Register backward hook only if tensor requires gradients
            if output.requires_grad:
                def backward_hook(grad):
                    storage_dict[name]['grad'] = grad.detach()
                output.register_hook(backward_hook)

        elif isinstance(output, (tuple, list)):
            # Detach each tensor element
            storage_dict[name]['activation'] = [o.detach() for o in output if isinstance(o, torch.Tensor)]

            # Register backward hooks on each tensor
            grads = [None] * len(output)
            def make_bwd_hook(i):
                def backward_hook(grad):
                    grads[i] = grad.detach()
                    storage_dict[name]['grad'] = grads
                return backward_hook

            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor) and o.requires_grad:
                    o.register_hook(make_bwd_hook(i))

        else:
            # Fallback: unknown type
            storage_dict[name]['activation'] = output

    return forward_hook


def register_all_hooks(model, storage_dict):
    """
    Registers the forward-backward hooks to all leaf layers of a model.
    """
    for name, layer in model.named_modules():
        if len(list(layer.children())) == 0:  # Only leaf layers
            layer.register_forward_hook(forward_backward_hook(name, storage_dict))

def get_dict_key(dict_, val):
    return list(dict_.keys())[np.where(np.array(list(dict_.values())) == val)[0][0]]