<div align="center">

<span style="font-size:12px">Coreset Learning Based Sparse Black-box Adversarial Attack For Video Recognition</span> </h2> 

<div>
    <a target='_blank'>Jiefu Chen </a>&emsp;
    <a target='_blank'>Tong Chen </a>&emsp;
    <a target='_blank'>Xing Xu  </a>&emsp;
    <a target='_blank'>Guoqing Wang </a>&emsp;
    <a target='_blank'>Yang Yang  </a>&emsp;
    <a target='_blank'>Heng Tao Shen </a>&emsp;

</div>
<br>
<div>
    University of Electronic Science and technology
</div>

<br>
<img src="./docs/static/images/image.png?raw=true" width="768px">

<div align="justify"> we propose a novel frame selection algorithm named CLVA, which is based on the coreset concept of active learning, to address the issues of frame selection efficiency and sparsity in existing video attack models. Specifically, CLVA simulates the process of coreset learning to identify frames with high weight for the attack recognition model. To achieve this, we consider a complete video clip as a dataset and treat all frames within it as a mini-dataset. We combine the coreset search algorithm with the attack and recognition algorithm to find frames with higher weights. The frames with the shortest distance are then extracted as coreset members using the K-Center-Greedy algorithm. We guarantee that the frames we find are optimal solutions by accurately calculating the distance between frames through the feature representation of the model. Finally, we perform a sparse attack on the selected frames.
</div>
<br>


</div>


## Environment
```
git clone https://github.com/machineNo6/CLVA.git
cd CLVA
conda create -n CLVA python=3.7
conda activate CLVA

pip install -r requirements.txt
```

## Quick start

#### Pretrained Models
Please download our [pre-trained models](https://pan.baidu.com/s/133O8LhydB9H13I3LbI12uA?pwd=qnbw ) (Extraction Code: qnbw, I3D+HMDB51 model) and put them in `./checkpoints`.

The models for the remaining combinations need to be downloaded on your own, or you can contact us to provide them for you.

#### Dataset

You need to download the data for the UCF-101, HMDB-51, and Kinetics-400 datasets yourself (downloading just one of them is acceptable).

You can find them at the following links:
[UCF-101](https://tensorflow.google.cn/datasets/catalog/ucf101)
[HMDB-51](https://pytorch.org/vision/stable/generated/torchvision.datasets.HMDB51.html)
[Kinetics-400](https://www.deepmind.com/open-source/kinetics)


#### Sparse video attack

```
python new_coreset.py 6 --gpus 0
```


## Acknowledgement
Thanks to
[DeepSAVA](https://github.com/TrustAI/DeepSAVA),
[Heuristic](https://github.com/zhipeng-wei/Heuristic_black_box_adversarial_attack_on_video_recognition_models), 
[SVA](https://github.com/FenHua/SVA), 
[AST](https://github.com/deepsota/astfocus),
[Coreset](https://github.com/ozansener/active_learning_coreset),
for sharing their code.


## Related Work
- [DeepSAVA: Sparse Adversarial Video Attack with Spatial Transformation](https://github.com/TrustAI/DeepSAVA)
- [Reinforcement Learning Based Sparse Black-box Adversarial Attack on Video Recognition Models](https://github.com/Doubiiu/CodeTalker)
- [Heuristic Black-box Adversarial Attacks on Video Recognition Models](https://github.com/zhipeng-wei/Heuristic_black_box_adversarial_attack_on_video_recognition_models)
- [Sparse Black-box Video Attack with Reinforcement Learning](https://github.com/FenHua/SVA)
- [Efficient Robustness Assessment via Adversarial Spatial-Temporal Focus on Videos](https://github.com/deepsota/astfocus)
- [Active Learning for Convolutional Neural Networks: A Core-Set Approach](https://github.com/ozansener/active_learning_coreset)



