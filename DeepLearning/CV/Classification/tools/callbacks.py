import torch

class SaveModelCallbacks():
    def on_epoch_end(self, model, epoch):
        torch.save(model.state_dict(), f'./Training_model_{epoch}.pth')
