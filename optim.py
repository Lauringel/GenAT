import torch
from torch import nn

# we create the following wrapper class to comfortably use torch.optim
class WrapperNet(nn.Module):
    def __init__(self, img, vqgan):
        super().__init__()
        
        # Keep original image for AT_loss.
        self.img = img

        # Normalize only for VQGAN initialization.
        img_norm = img * 2 - 1
        
        # --- initialize latent z ---
        h = vqgan.quant_conv(vqgan.encoder(img_norm))
        self.z = torch.nn.parameter.Parameter(
            data=h.clone().detach(),
            requires_grad=True,
        )
        
        # --- initialize logits s for v ---
        self.s = torch.nn.parameter.Parameter(
            data=torch.zeros(
                (img.shape[0], 1, img.shape[2], img.shape[3]),
                device=img.device,
            ),
            requires_grad=True,
        )

    def forward(self):
        return [self.s, self.z]

# optimization of z and s
def train(model, loss_function, iters=500, lr_z=0.005, lr_v=0.1):
    opt = torch.optim.Adam([
            {"params": model.z, "lr": lr_z},
            {"params": model.s, "lr": lr_v}
        ])
    for _ in range(iters):
            opt.zero_grad()
            loss = loss_function(model.img, model())
            loss.backward()
            opt.step()
    return model
