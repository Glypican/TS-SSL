"""
Task-Specific Self-Supervised Learning (TS-SSL) Autoencoder implementation.

This module implements the Spatial-Channel Attention Autoencoder (scAE)
based on the paper "Evaluation of a Task-Specific Self-Supervised Learning 
Framework in Digital Pathology Relative to Transfer Learning Approaches 
and Existing Foundation Models".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union


class SpatialAttention(nn.Module):
    """
    Spatial attention module for capturing contextual relations among local areas.
    
    This module implements the spatial attention mechanism from the paper,
    using a non-local operation self-attention to capture context relations
    between different local features.
    """
    
    def __init__(self, in_channels: int):
        """
        Initialize spatial attention module.
        
        Args:
            in_channels: Number of input channels
        """
        super().__init__()
        
        # Reduced channel dimension for attention computation
        self.reduced_channels = max(in_channels // 8, 1)
        
        # Define query, key, value projections
        self.query_conv = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Final 1x1 convolution for output
        self.output_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.output_conv.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spatial attention module.
        
        Args:
            x: Input feature map (B, C, H, W)
            
        Returns:
            Feature map with contextual relation information (B, C, H, W)
        """
        batch_size, channels, height, width = x.size()
        
        # Reshape for attention computation
        spatial_size = height * width
        
        # Generate query, key, value projections
        query = self.query_conv(x).view(batch_size, self.reduced_channels, spatial_size)
        key = self.key_conv(x).view(batch_size, self.reduced_channels, spatial_size)
        value = self.value_conv(x).view(batch_size, channels, spatial_size)
        
        # Transpose for matrix multiplication
        query = query.permute(0, 2, 1)  # B, HW, C'
        
        # Calculate attention map
        attention = torch.bmm(query, key)  # B, HW, HW
        attention = F.softmax(attention / (self.reduced_channels ** 0.5), dim=-1)  # Scale dot-product
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, HW
        out = out.view(batch_size, channels, height, width)  # B, C, H, W
        
        # Final convolution
        out = self.output_conv(out)
        
        # Add residual connection
        return out + x


class ChannelAttention(nn.Module):
    """
    Channel attention module for focusing on informative feature channels.
    
    This module implements the channel attention mechanism from the paper,
    using global average pooling and a gating mechanism to weigh channels.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Initialize channel attention module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel dimension in FC layers
        """
        super().__init__()
        
        # Ensure reduction_ratio doesn't make channels too small
        reduced_channels = max(in_channels // reduction_ratio, 8)
        
        # Global average pooling is implicit in forward
        
        # Multi-layer perceptron for channel attention
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the channel attention module.
        
        Args:
            x: Input feature map (B, C, H, W)
            
        Returns:
            Channel-weighted feature map (B, C, H, W)
        """
        batch_size, channels, height, width = x.size()
        
        # Global average pooling to get channel-wise weights
        y = x.view(batch_size, channels, -1).mean(dim=2)  # B, C
        
        # Apply MLP to get attention weights
        y = self.mlp(y)  # B, C
        
        # Apply channel attention
        y = y.view(batch_size, channels, 1, 1)  # B, C, 1, 1
        
        # Element-wise multiplication for channel attention
        return x * y.expand_as(x)


class SpatialChannelAttentionBlock(nn.Module):
    """
    Combined spatial and channel attention block.
    
    This block applies both spatial and channel attention in sequence
    to enhance feature representations.
    """
    
    def __init__(self, in_channels: int):
        """
        Initialize spatial-channel attention block.
        
        Args:
            in_channels: Number of input channels
        """
        super().__init__()
        
        self.spatial_attention = SpatialAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spatial-channel attention block.
        
        Args:
            x: Input feature map (B, C, H, W)
            
        Returns:
            Enhanced feature map (B, C, H, W)
        """
        # Apply spatial attention
        x_spatial = self.spatial_attention(x)
        
        # Apply channel attention
        x_channel = self.channel_attention(x)
        
        # Combine spatial and channel attention
        # Concatenate and apply 1x1 conv to fuse them
        return x_spatial + x_channel


class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization and activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        use_activation: bool = True,
    ):
        """
        Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride of convolution
            padding: Padding size
            use_bn: Whether to use batch normalization
            use_activation: Whether to use activation function
        """
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        if use_activation:
            layers.append(nn.ReLU(inplace=True))
            
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional block."""
        return self.conv_block(x)


class EncoderBlock(nn.Module):
    """
    Encoder block with convolution and attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
    ):
        """
        Initialize encoder block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_attention: Whether to use spatial-channel attention
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        # Convolutional layers
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        
        # Attention block
        if use_attention:
            self.attention = SpatialChannelAttentionBlock(out_channels)
        
        # Downsampling layer
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (downsampled_features, skip_connection_features)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        # Store for skip connection
        skip = x
        
        # Downsample
        x = self.pool(x)
        
        return x, skip


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
    ):
        """
        Initialize decoder block.
        
        Args:
            in_channels: Number of input channels
            skip_channels: Number of channels from skip connection
            out_channels: Number of output channels
            use_attention: Whether to use spatial-channel attention
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        # Upsampling layer
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        # Convolutional layers
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        
        # Attention block
        if use_attention:
            self.attention = SpatialChannelAttentionBlock(out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input tensor
            skip: Skip connection tensor from encoder
            
        Returns:
            Upsampled and processed tensor
        """
        # Upsample
        x = self.upsample(x)
        
        # Ensure dimensions match for concatenation
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        return x


class SpatialChannelAutoEncoder(nn.Module):
    """
    Spatial-Channel Attention Autoencoder (scAE) for Task-Specific Self-Supervised Learning.
    
    This is the main implementation of the TS-SSL autoencoder described in the paper,
    with spatial and channel attention mechanisms integrated into a U-Net-like architecture.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dims: List[int] = [64, 128, 256, 512],
        latent_dim: int = 512,
        output_channels: Optional[int] = None,
        use_attention: bool = True,
    ):
        """
        Initialize the Spatial-Channel Attention Autoencoder.
        
        Args:
            input_channels: Number of input image channels
            hidden_dims: List of hidden dimensions for each layer
            latent_dim: Dimension of the latent space
            output_channels: Number of output channels (defaults to input_channels)
            use_attention: Whether to use attention mechanisms
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels or input_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        
        # First encoder block (input -> first hidden dim)
        self.encoder_blocks.append(
            EncoderBlock(input_channels, hidden_dims[0], use_attention=use_attention)
        )
        
        # Remaining encoder blocks
        for i in range(len(hidden_dims) - 1):
            self.encoder_blocks.append(
                EncoderBlock(hidden_dims[i], hidden_dims[i+1], use_attention=use_attention)
            )
        
        # Bottleneck block
        self.bottleneck = nn.Sequential(
            ConvBlock(hidden_dims[-1], latent_dim),
            ConvBlock(latent_dim, hidden_dims[-1])
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        # Decoder blocks (in reverse order)
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_blocks.append(
                DecoderBlock(hidden_dims[i], hidden_dims[i-1], hidden_dims[i-1], use_attention=use_attention)
            )
        
        # Final decoder block
        self.final_conv = ConvBlock(hidden_dims[0], self.output_channels, use_bn=False)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode input into latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (bottleneck_features, skip_connections)
        """
        skips = []
        
        # Apply encoder blocks and store skip connections
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skips.append(skip)
        
        # Apply bottleneck
        x = self.bottleneck(x)
        
        return x, skips
    
    def decode(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode latent representation back to image space.
        
        Args:
            x: Latent representation
            skips: Skip connections from encoder
            
        Returns:
            Reconstructed image
        """
        # Reverse skip connections for decoder
        skips = skips[::-1]
        
        # Apply decoder blocks with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, skips[i])
        
        # Final convolution to get output channels
        x = self.final_conv(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed tensor
        """
        # Encode
        x, skips = self.encode(x)
        
        # Decode
        x = self.decode(x, skips)
        
        return x
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation of input.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        # Encode without skip connections
        latent, _ = self.encode(x)
        return latent


class TS_SSL_Encoder(nn.Module):
    """
    Task-Specific Self-Supervised Learning Encoder.
    
    This module extracts the encoder part of the scAE for use in downstream tasks.
    """
    
    def __init__(self, autoencoder: SpatialChannelAutoEncoder, return_features: bool = False):
        """
        Initialize the TS-SSL encoder.
        
        Args:
            autoencoder: Trained scAE model
            return_features: Whether to return features instead of latent representation
        """
        super().__init__()
        
        self.input_channels = autoencoder.input_channels
        self.hidden_dims = autoencoder.hidden_dims
        self.latent_dim = autoencoder.latent_dim
        self.use_attention = autoencoder.use_attention
        self.return_features = return_features
        
        # Copy encoder blocks
        self.encoder_blocks = autoencoder.encoder_blocks
        
        # Copy bottleneck
        self.bottleneck = autoencoder.bottleneck
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation or tuple of (latent, features) if return_features=True
        """
        features = []
        
        # Apply encoder blocks and store features
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            features.append(skip)
        
        # Apply bottleneck
        latent = self.bottleneck(x)
        
        if self.return_features:
            return latent, features
        
        return latent
