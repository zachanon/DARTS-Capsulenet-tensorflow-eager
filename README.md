# DARTS-Capsules

Very simple implementation of the DARTS algorithm used to find an optimized CNN block to feed to the dynamic routing algorithm.


## Notes

Project done while exploring Capsule Routing and DARTS algorithm, years ago now when both of these were quite new. DARTS since then has had many improvements made[1][2] so it's best to regard this repository as a historical example. As far as I know this was the earliest implementation of the DARTS algorithm in Tensorflow.

## Usage

This repository can be conceptually divided into two segments. The DARTS algorithm[3] is used to create a 'feature extractor' which is fed into a Capsule Routing[4] system. If you would like to apply the DARTS optimization to find a feature extractor for a different general architecture, simply clone the operations.py file and the ConvolutionalBlock class found in models.py. If you would like to test the dynamic routing algorithm, the code is found in capsules.py and class CapsuleBlock in models.py. The code found in this repository has not been tested to confirm that it reaches convergence, only that gradients flow and losses decrease, so your milage may vary.

[1]: https://arxiv.org/abs/1907.05737
[2]: https://arxiv.org/abs/1909.06035
[]: https://arxiv.org/abs/1806.09055
[]: https://arxiv.org/abs/1710.09829 
