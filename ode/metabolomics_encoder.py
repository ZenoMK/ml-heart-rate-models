from ode.modules_cnn import CNNEncoder
import torch.nn as nn

class MetabolomicsEncoder(nn.Module):
    def __init__(self,
                 metabolomics_encoder_input_dim,
                 metabolomics_encoder_hidden_dim,
                 metabolomics_encoder_output_dim,
                 ):
        super(MetabolomicsEncoder, self).__init__()

        self.metabolomics_encoder_input_dim = metabolomics_encoder_input_dim
        self.metabolomics_encoder_hidden_dim = metabolomics_encoder_hidden_dim
        self.metabolomics_encoder_output_dim = metabolomics_encoder_output_dim



        self.metabolomics_encoder = nn.Sequential(
            nn.Linear(self.metabolomics_encoder_input_dim, self.metabolomics_encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.metabolomics_encoder_hidden_dim, self.metabolomics_encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.metabolomics_encoder_hidden_dim, self.metabolomics_encoder_output_dim),
            nn.ReLU()
        )

    def forward(self, metabolomics_data):
        """
        Forward pass for metabolomics data

        Args:
            metabolomics_data: Tensor of shape (batch_size, metabolomics_encoder_input_dim)
                              Contains metabolomics measurements

        Returns:
            embedding: Tensor of shape (batch_size, metabolomics_encoder_output_dim)
        """
        return self.metabolomics_encoder(metabolomics_data)

