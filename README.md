# CoCo-teaching
Proceedings of the International Conference on Multimedia and Expo (ICME-2024).


## Abstract 
Noisy labels in datasets may cause deep neural networks (DNNs) to memorize misleading information, thereby impacting generalization performance. Therefore, learning with noisy labels holds significant practical importance. Small-loss sample selection methods often underutilize available data and constrain the model's potential, especially in the presence of complex or high-ratio noisy labels. In this paper, we propose Consensus Co-teaching (CoCo-teaching), introducing a consensus loss that operates on all data samples for additional supervision. This promotes the convergence of the two models towards greater similarity. Furthermore, a dynamic learning scheme is implemented to resist the memorization effects of DNNs, where models progressively rely on their consensus instead of the noisy labels during training. Meanwhile, we leverage the flip semantic consistency of images to enhance model divergence. Extensive experiments demonstrate the superiority of CoCo-teaching on synthetic and real-world noisy datasets.
