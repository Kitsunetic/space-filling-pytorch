# space-filling-pytorch

A set of GPU-optimized space-filling-curve algorithms (e.g., Hilbert, Z-order) implemented in PyTorch and [OpenAI Triton](https://github.com/triton-lang/triton).
These algorithms are more than 100 times faster than typical PyTorch-based implementations due to kernel fusion.
This library is particularly useful for handling point clouds in PyTorch deep learning networks, such as [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3).


## Usage

```py
import torch
from space_filling_pytorch import hilbert_encode, z_order_encode

batch_size = 2
num_points = 16
xyz = torch.rand(batch_size, num_points, 3, device="cuda") * 2 - 1  # value range must be [-1, 1]

# Hilbert Curve with no transpose (xyz order)
hilbert_encode(xyz, space_size=1024, x_offset=0, y_offset=1, z_offset=2)
>>> tensor([[ 27008646,  174416382,   99184467,  545318174, 1014259428,  947817691,
             974147274,  465746813,  590435514,  122722004,  897101603,  535449979,
             520237253,  446529249,  197013177,  254906709],
            [551194570,  532868556,  350803712,  922818975,  602215252,  253605249,
             600868136,  557531360,  625006013,  678917039,  957204642,  187281814,
             621257029,  989726818,  615092084,  909388620]], device='cuda:0')

# Hilbert Curve with transpose (zyx order)
hilbert_encode(xyz, space_size=1024, x_offset=2, y_offset=1, z_offset=0)
>>> tensor([[  27203104,  943467886,   90203125,  276882718,  136738518,  171280879,
              211356654,  466326141,  322000058,   51565506,  933932599,  422793567,
              424300741,  480083681, 1001350699, 1026735333],
            [ 282759114,  433308876,  619239168,  903500185,  333779796, 1037547831,
              332432680,  289095904,  356570557,  687821371,  223340726, 1006162308,
              352821573,  217999680,  346656628,  921071576]], device='cuda:0')

# Z-Order with no transpose (xyz order)
z_order_encode(xyz, space_size=1024, x_offset=0, y_offset=1, z_offset=2)
>>> tensor([[ 44778821, 226778406, 121328308, 815839782, 539433139, 631067087,
             599439830, 397986105, 848144099,  29398278, 693723698, 344726626,
             335653660, 331813849, 210352353, 182524594],
            [819692704, 349517781, 507224630, 750011964, 841590237, 175939993,
             840220003, 882290987, 935427276, 948966804, 665137179, 203507248,
             923330332, 596839195, 871191299, 758935523]], device='cuda:0')
```


## Installation

To install the library, use the following command:

```sh
pip install git+https://github.com/Kitsunetic/space-filling-pytorch.git
```


## Further Improvements

Currently, this library works only with 3D point cloud with float32 dtype because these are all of what I needed.
If you need help, such as another input/output format or decoding, please contact me through [issues](https://github.com/Kitsunetic/space-filling-pytorch/issues) or email.

- [ ] Implement additional algorithms such as Peano, Moore, Gosper Codes.
- [ ] Extent algorithms for more general inputs (not just 3D point cloud, but also 1D, 2D, 4D cases)
- [ ] Fusion of space filling curve code generation / ordering / gathering processes.
