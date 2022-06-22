SiamCross基于Ocean框架，请先搭建好所有Ocean相关环境。

下为Ocean官方搭建流程。

> ### Ocean
>
> **[[Paper]](https://arxiv.org/abs/2006.10721) [[Raw Results]](https://drive.google.com/drive/folders/1w_SifcV_Ddu2TSqaV-14XSgLlKZq2lPN?usp=sharing) [[Training and Testing]](https://github.com/researchmm/TracKit/tree/master/lib/tutorial/Ocean/ocean.md) [[Demo]](https://www.youtube.com/watch?v=83-XCEsQ1Kg&feature=youtu.be)** <br/>
>
> Official implementation of the Ocean tracker. Ocean proposes a general anchor-free based tracking framework. It includes a pixel-based anchor-free regression network to solve the weak rectification problem of RPN, and an object-aware classification network to learn robust target-related representation. Moreover, we introduce an effective multi-scale feature combination module to replace heavy result fusion mechanism in recent Siamese trackers. An additional **TensorRT** toy demo is provided in this repo.

搭建完毕后，按如下流程：

1. 修改models/backbones 和 connect相关代码，将resnet启用最后一层。
2. 参考论文4.1节按需修改相关训练和推理参数。
3. 参考tracker/siamcross.py，将SCAN和PCA模块加入模型中。
4. 随后进行推理训练即可。

临近毕业整理代码成型时间过去一年多，有匆忙错误之处可联系作者。