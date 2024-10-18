# space-filling-pytorch

A set of GPU-optimized space-filling-curve algorithms (e.g., Hilbert, Z-order) implemented in PyTorch and [OpenAI Triton](https://github.com/triton-lang/triton).
These algorithms are more than 30 times faster than typical PyTorch-based implementations using kernel fusion.
This library is particularly useful for handling point clouds in PyTorch deep learning networks, such as [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3).


## Usage

### Usage for Batched Input

```py
import torch
from space_filling_pytorch import encode

batch_size = 2
num_points = 16
xyz = torch.rand(batch_size, num_points, 3, device="cuda")
xyz = xyz * 2 - 1  # coordination must be normalized into [-1, 1]

# Hilbert Curve with no transpose (xyz order)
encode(xyz, space_size=1024, method='hilbert', convention="xyz")
>>> tensor([[ 27008646,  174416382,   99184467,  545318174, 1014259428,  947817691,
             974147274,  465746813,  590435514,  122722004,  897101603,  535449979,
             520237253,  446529249,  197013177,  254906709],
            [551194570,  532868556,  350803712,  922818975,  602215252,  253605249,
             600868136,  557531360,  625006013,  678917039,  957204642,  187281814,
             621257029,  989726818,  615092084,  909388620]], device='cuda:0')

# Hilbert Curve with transpose (zyx order)
encode(xyz, space_size=1024, method='hilbert', convention="zyx")
>>> tensor([[  27203104,  943467886,   90203125,  276882718,  136738518,  171280879,
              211356654,  466326141,  322000058,   51565506,  933932599,  422793567,
              424300741,  480083681, 1001350699, 1026735333],
            [ 282759114,  433308876,  619239168,  903500185,  333779796, 1037547831,
              332432680,  289095904,  356570557,  687821371,  223340726, 1006162308,
              352821573,  217999680,  346656628,  921071576]], device='cuda:0')

# Z-Order with no transpose (xyz order)
encode(xyz, space_size=1024, method='z', convention="xyz")
>>> tensor([[ 44778821, 226778406, 121328308, 815839782, 539433139, 631067087,
             599439830, 397986105, 848144099,  29398278, 693723698, 344726626,
             335653660, 331813849, 210352353, 182524594],
            [819692704, 349517781, 507224630, 750011964, 841590237, 175939993,
             840220003, 882290987, 935427276, 948966804, 665137179, 203507248,
             923330332, 596839195, 871191299, 758935523]], device='cuda:0')
             
# Assigning first 16bits for batch index (default False)
encode(xyz, space_size=1024, method='z', convention="xyz", assign_batch_index=True)
>>> tensor([[140119069688320, 140119069688328, 140119069688336, 140119069688344,
            140119069688352, 140119069688360, 140119069688368, 140119069688376,
            140119069688384, 140119069688392, 140119069688400, 140119069688408,
            140119069688416, 140119069688424, 140119069688432, 140119069688440],
           [140119069688448, 140119069688456, 140119069688464, 140119069688472,
            140119069688480, 140119069688488, 140119069688496, 140119069688504,
            140119069688512, 140119069688520, 140119069688528, 140119069688536,
            140119069688544, 140119069688552, 140119069688560, 140119069688568]],
          device='cuda:9')

```

### Usage for Unpadded Input

This library can process unpadded input with Flash-Attention format that flattened batch with seqlen and max_seqlen.

```py
import torch
from space_filling_pytorch import encode_unpadded

xyz = torch.rand(1000, 3, device="cuda")
xyz = xyz * 2 - 1  # coordination must be normalized into [-1, 1]
seqlen = torch.tensor([0, 100, 500, 1000], dtype=torch.int32, device="cuda")
max_seqlen = 500

# We can apply unpadded input with the same way
encode_unpadded(xyz, space_size=1024, method="hilbert", convention="xyz", assign_batch_index=True).shape
>>> torch.Size([1000])
```


## Installation

To install the library, use the following command:

```sh
pip install git+https://github.com/Kitsunetic/space-filling-pytorch.git
```


## Further Improvements

Currently, this library works only with 3D point cloud with float32 dtype because these are all of what I needed.
If you need help, such as another input/output format or decoding, please contact me through [issues](https://github.com/Kitsunetic/space-filling-pytorch/issues) or email.

- [x] Performance comparison with this library to existing implementations.
- [x] Support for non-contiguous input tensors.
- [x] Support for Flash-Attention-like unpadded input (2024.10.18).
- [ ] Implement additional algorithms such as Peano, Moore, Gosper Codes.
- [ ] Extent algorithms for more general inputs (not just 3D point cloud, but also 1D, 2D, 4D cases).
- [ ] Kernel fusion of space filling curve code generation / ordering / gathering processes.
