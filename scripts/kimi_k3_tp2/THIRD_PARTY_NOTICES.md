# Third-party notices

These tools are source code; they do not redistribute Kimi K3 model weights.
Generated rank-local checkpoints remain subject to the license of the source
checkpoint.

## Kimi K3 model

The conversion contract targets
[`kernelpool/Kimi-K3-2bit-UVMAX`](https://huggingface.co/kernelpool/Kimi-K3-2bit-UVMAX)
at revision `edb5113218df612f4a92f95145680f3f8eacd375`. The model repository
declares the Kimi K3 License. Review the
[official Kimi K3 license](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE)
before downloading or using the model.

`k3_tp_checkpoint.py` requires `LICENSE`, `LICENSE.md`, or `LICENSE.txt` in the
input metadata directory and copies that file byte-for-byte into both generated
rank checkpoints. Do not remove it when distributing a converted checkpoint.

## MLX-LM

The rank-local load sequence and Kimi K3 sharding contract are derived from
[MLX-LM pull request 1626](https://github.com/ml-explore/mlx-lm/pull/1626) at
commit `7d505c285b801108a52c23353c7fb6af07204717`.

MLX-LM is distributed under the MIT License:

> Copyright (c) 2023 Apple Inc.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

See the [upstream MLX-LM license](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE).

## EXO

EXO is distributed under the Apache License 2.0. See the repository root
license for its terms.
