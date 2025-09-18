# PINNs-Torch Components Documentation

## Activations (from activations.py)

| Name | Description | Parameters |
|------|-------------|------------|
| SINE | Sine activation | `omega=30` (float): Frequency parameter |
| FINER | Finer activation | `omega=30` (float): Frequency parameter |
| GAUSSIAN | Gaussian activation | `sigma=10` (float): Standard deviation |
| GAUSSIAN_FINER | Finer Gaussian activation | `sigma=10` (float), `omega=30` (float) |
| WIRE | WIRE activation | `omega_0=20` (float), `sigma_0=10` (float) |
| WIRE_FINER | Finer WIRE activation | `omega_0=20` (float), `sigma_0=10` (float), `omega_finer=5` (float) |
| HOSC | Hyperbolic sine activation | `beta=10` (float): Scaling parameter |
| HOSC_FINER | Finer hyperbolic sine | `beta=10` (float), `omega=30` (float) |
| SINC | Sinc activation | `omega=30` (float): Frequency parameter |
| RELU | ReLU activation | None |
| LEAKY_RELU | Leaky ReLU activation | `negative_slope=0.01` (float) |
| SILU | Sigmoid Linear Unit | None |
| SIGMOID | Sigmoid activation | None |
| TANH | Hyperbolic tangent | None |

## Initializers (from initializations.py)

| Name | Description | Parameters |
|------|-------------|------------|
| SIREN | SIREN-specific initialization | `in_features` (int), `is_first=True` (bool), `omega=30` (float) |
| FINER | FINER-specific initialization | `in_features` (int), `is_first=False` (bool), `omega=30` (float), `k=1` (int) |
| XAVIER_UNIFORM_FINER | Xavier uniform with finer scale | `gain=1.0` (float), `k=1` (float) |
| XAVIER_NORM_FINER | Xavier normal with finer scale | `gain=1.0` (float), `k=1` (float) |
| XAVIER_UNIFORM | Xavier uniform initialization | `gain=1.0` (float) |
| XAVIER_NORMAL | Xavier normal initialization | `gain=1.0` (float) |
| NORMAL | Normal distribution | `mean=0.0` (float), `std=0.1` (float) |
| UNIFORM | Uniform distribution | `a=-0.1` (float), `b=0.1` (float) |
| IDENTITY | Identity initialization | None |
| ORTHOGONAL | Orthogonal initialization | `gain=1.0` (float) |

## Layers (from layers.py)

### 1. BaseLayer
- **Description**: Base MLP layer with custom initializer and activation
- **Parameters**:
  - `in_features` (int): Input dimension
  - `out_features` (int): Output dimension
  - `initializer` (str): Name of weight initializer
  - `activation` (str): Name of activation function
  - `is_last` (bool, optional): Whether this is the last layer. Default: False
  - `initializer_kwargs` (dict, optional): Arguments for initializer
  - `activation_kwargs` (dict, optional): Arguments for activation
  - `dtype` (torch.dtype, optional): Data type for the layer

### 2. ResidualBaseBlock
- **Description**: Residual block with custom initializer and activation
- **Parameters**:
  - `in_features` (int): Input dimension
  - `out_features` (int): Output dimension
  - `initializer` (str): Name of weight initializer
  - `activation` (str): Name of activation function
  - `residual_weight` (float, optional): Weight for residual connection. Default: None
  - `initializer_kwargs` (dict, optional): Arguments for initializer
  - `activation_kwargs` (dict, optional): Arguments for activation
