# ConvNeXt implimentaion written from scratch in PyTorch

Model parameters and architecture based on the whitepaper `"A ConvNet for the 2020s"` by Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, & Saining Xie at Facebook AI Research (FAIR) together with UC Berkeley.

Link to the paper:
https://arxiv.org/pdf/2201.03545.pdf

# Summary
In 2021 the
introduction of Vision Transformers (ViTs) quickly
superseded ConvNets as the state-of-the-art image classification model. However, a vanilla ViT faces difficulties
when applied to general computer vision tasks such as object
detection and semantic segmentation. It is the hierarchical
Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers practically viable
as a generic vision backbone and demonstrating remarkable
performance on a wide variety of vision tasks. Nonetheless,
the effectiveness of such hybrid approaches is still largely
credited to the intrinsic superiority of Transformers, rather
than the inherent inductive biases of convolutions. 

In `"A ConvNet for the 2020s"`, researchers reexamined the design spaces and test the limits of
what a pure ConvNet can achieve. They gradually “modernized”
a standard ResNet toward the design of a vision Transformer,
and discovered several key components that contribute to the
performance difference along the way. The outcome of this
exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules,
ConvNeXts compete favorably with Transformers in terms of
accuracy and scalability, achieving 87.8% ImageNet top-1
accuracy and outperforming Swin Transformers on COCO
detection and ADE20K segmentation, while maintaining the
simplicity and efficiency of standard ConvNets.


# ImageNet-1K classification results for ConvNets, vision Transformers, and ConvNeXt
![alt text](docs/imagenet-1k.png)

<br />

# Pre-Training Config for ConvNeXt-T/S/B/L/XL
| | |
|-------------|-------------|
| `optimizer`  | AdamW |
| `base learning rate`    | 5e-5 |
| `weight decay`  | 1e-8 |
| `optimizer momentum`      | β<sub>1</sub>, β<sub>2</sub> = 0.9, 0.999|
| `batch size`  | 512 |
| `training epochs`       | 30|
| `learning rate schedule`  | cosine decay|
| `layer-wise lr decay`  | 0.8 |
| `warmup epochs` | None|
| `warmup schedule` | N/A |
| `randaugment` | (9, 0.5) |
|  `mixup` |  None |
| `cutmix`    | None|
| `random erasing`| 0.25 |
|  `label smoothing`| 0.1|
|  `stochastic depth`|  0.0 / 0.1 / 0.2 / 0.3 / 0.4|
|  `layer scale`| pre-trained |
| `head init scale`| 0.001 |
| `gradient clip`| None |
| `exp. mov. agv. (EMA)`| None(T-L) / 0.9999(XL)|
