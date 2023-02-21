from torch import optim
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

from tqdm import tqdm

from training_utils import build_3d_model, read_config, build_2d_model
from acdc_dataset import create_3d_acdc_dataset, create_2d_acdc_dataset

warnings.filterwarnings("ignore", category=UserWarning)

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1).cuda())

    def forward(self, x):
        out = self.model(x)
        logits = out['pred'][-1]
        scaled_logits = self.temperature_scale(logits)

        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(torch.max(F.softmax(logits, dim=1), dim=1)[0].cpu()[0, 3], cmap='plasma')
        #ax[1].imshow(torch.max(F.softmax(scaled_logits, dim=1), dim=1)[0].cpu()[0, 3], cmap='plasma')
        #plt.show()

        return scaled_logits

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return torch.div(logits, self.temperature)

    def set_temperature(self, val_loader):
            """
            Tune the tempearature of the model (using the validation set).
            We're going to set it to optimize NLL.
            valid_loader (DataLoader): validation set loader
            """
            nll_criterion = nn.CrossEntropyLoss()
            ece_criterion = _ECELoss()

            # First: collect all the logits and labels for the validation set
            logits_list = []
            labels_list = []
            for data in tqdm(val_loader):
                x, y_true = data['x'], data['y']
                y_true = torch.argmax(y_true, dim=1)
                self.model.eval()
                with torch.no_grad():
                    out = self.model(x)
                    logits = out['pred'][-1]
                    logits_list.append(logits)
                    labels_list.append(y_true)
            logits = torch.cat(logits_list).to('cuda:0')
            labels = torch.cat(labels_list).to('cuda:0')
            
            # Calculate NLL and ECE before temperature scaling
            before_temperature_nll = nll_criterion(logits, labels).item()
            before_temperature_ece = ece_criterion(logits, labels).item()
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(self.temperature_scale(logits), labels)
                loss.backward()
                return loss
            optimizer.step(eval)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            print('Optimal temperature: %.3f' % self.temperature.item())
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))


def get_temperature(dimensions):
    img_size = 128
    if dimensions == 3:
        config = read_config('3d_model_128_3\config.yaml')
        train_path = 'ACDC_resampled_cropped_4/*'
        val_path = 'ACDC_resampled_cropped_4/*'
        dataloaders = create_3d_acdc_dataset(train_path=train_path, 
                                            val_path=val_path,
                                            batch_size=config['batch_size'],
                                            device=config['device'],
                                            binary=config['binary'],
                                            val_subset_size=config['val_subset_size'],
                                            data_augmentation_utils=None,
                                            img_size=img_size)

        orig_model = build_3d_model(config)
        orig_model.load_state_dict(torch.load('3d_model_128_3\weights.pth'))
        model = ModelWithTemperature(orig_model)

        model.set_temperature(dataloaders['val_dataloader'])
        Path('out_temperature_3d_128').mkdir(parents=True, exist_ok=True)
        model_filename = os.path.join('out_temperature_3d_128', 'model_with_temperature.pth')

        Path("out_temperature").mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), model_filename)
        print('Temperature scaled model saved to %s' % model_filename)
        print('Done!')
    elif dimensions == 2:
        config = read_config('2d_model_128_2\config.yaml')
        train_path = 'ACDC_resampled_cropped_2/*'
        val_path = 'ACDC_resampled_cropped_2/*'
        dataloaders = create_2d_acdc_dataset(train_path=train_path, 
                                    val_path=val_path,
                                    batch_size=config['batch_size'],
                                    device=config['device'],
                                    val_subset_size=config['val_subset_size'],
                                    data_augmentation_utils=None,
                                    img_size=img_size)

        orig_model = build_2d_model(config)
        orig_model.load_state_dict(torch.load('2d_model_128_2\weights.pth'))
        model = ModelWithTemperature(orig_model)

        model.set_temperature(dataloaders['val_dataloader'])
        Path('out_temperature_2d_128').mkdir(parents=True, exist_ok=True)
        model_filename = os.path.join('out_temperature_2d_128', 'model_with_temperature.pth')

    torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model saved to %s' % model_filename)
    print('Done!')

#get_temperature(dimensions=2)