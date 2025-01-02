import torch
import config as config
from Loss.custom_loss import MyLoss



class RhythmNetLoss(torch.nn.Module):
    def __init__(self, weight=150.0):
        super(RhythmNetLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.lambd = weight
        self.gru_outputs_considered = None
        self.custom_loss = MyLoss()
        self.device = config.DEVICE

    def forward(self, gru_outputs, target):
        l1_loss = self.l1_loss(gru_outputs, target)
        smooth_loss_component = self.smooth_loss(gru_outputs[:6])

        loss = l1_loss + self.lambd*smooth_loss_component
        return loss

    # Need to write backward pass for this loss function
    def smooth_loss(self, gru_outputs):
        smooth_loss = torch.zeros(1).to(device=self.device)
        self.gru_outputs_considered = gru_outputs.flatten()
        # hr_mean = self.gru_outputs_considered.mean()
        for hr_t in self.gru_outputs_considered:
            # custom_fn = MyLoss.apply
            smooth_loss = smooth_loss + self.custom_loss.apply(torch.autograd.Variable(hr_t, requires_grad=True),
                                                               self.gru_outputs_considered,
                                                               self.gru_outputs_considered.shape[0])
        return smooth_loss / self.gru_outputs_considered.shape[0]