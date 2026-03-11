"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Store original shape
        orig_shape = volume.shape

        # Pad y and z dimensions to patch_size (64x64) using zero-padding
        new_shape = (orig_shape[0], self.patch_size, self.patch_size)
        padded_volume = np.zeros(new_shape)
        padded_volume[:orig_shape[0],
                      :orig_shape[1],
                      :orig_shape[2]] = volume

        # Run inference on padded volume slice by slice along axis 0
        prediction = np.zeros(new_shape)

        for slc_ix in range(padded_volume.shape[0]):
            slc = padded_volume[slc_ix, :, :]
            slc_tensor = torch.from_numpy(slc[None, None, :, :]).float().to(self.device)

            with torch.no_grad():
                output = self.model(slc_tensor)

            pred_label = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            prediction[slc_ix, :, :] = pred_label

        # Crop back to original dimensions
        return prediction[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        slices = np.zeros(volume.shape)

        for slc_ix in range(volume.shape[0]):
            slc = volume[slc_ix, :, :]
            slc_tensor = torch.from_numpy(slc[None, None, :, :]).float().to(self.device)

            with torch.no_grad():
                prediction = self.model(slc_tensor)

            pred_label = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
            slices[slc_ix, :, :] = pred_label

        return slices
