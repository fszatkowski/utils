import torch

from matplotlib import pyplot as plt


def plot_grad_flow(model: torch.nn.Module) -> None:
    """
    Creates figure visualizing average gradient values in the network layers.

    Example usage in the training loop:

    plt.figure()
    for iteration in training_loop:
        ...
        # training code
        ...
        
        loss.backward()
        plot_grad_flow(model)
        optimizer.step()
        ....


    :param model: Any torch model.
    """
    named_parameters = model.named_parameters()

    ave_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
