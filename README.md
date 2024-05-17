# Barlow Twins Guided Filter(BTGF)
This is code for [*Upper Bounding Barlow Twins: A Novel Filter for Multi-relational Clustering*](https://arxiv.org/abs/2312.14066) AAAI-24.

Overall node clustering result:

![image-20231228103143576](https://s2.loli.net/2023/12/28/aQPhOAjN6sxSb74.png)


## Datasets
The statistics of the datasets are as follows:

<img src="https://s2.loli.net/2023/12/28/n4myzZlYp7ONet5.png" alt="image-20231228103516094" style="zoom:80%;" />

*DBLP* and *Amazon* can be found on [Google Drive](https://drive.google.com/drive/folders/1Ii2bpwZJSSkasi9IFGFkh1ZF-4gPWzUO?usp=drive_link).

## Usage
The `learning rate` and `weight decay` of the optimizer are set to $1e^{−2}$ and $1e^{−3}$. 

The filter’s parameters $k$ and $\alpha$ are tuned in $[1, 2, 3, 4]$ and $[1, 10, 100, 1000]$, respectively.

You can run BTGF with commands in the `script.sh`

```shell
python main.py -dataset ACM -epoch 400 -lr 1e-2 -wd 1e-3 -k 4 -a 10

python main.py -dataset amazon -epoch 400 -lr 1e-2 -wd 1e-3 -k 2 -a 1

python main.py -dataset aminer -epoch 400 -lr 1e-2 -wd 1e-3 -k 3 -a 100

python main.py -dataset DBLP_L -epoch 400 -lr 1e-2 -wd 1e-3 -k 2 -a 1000
```

## BibTex
Please cite our paper if you found our datasets or code helpful.
```
@inproceedings{qian2024upper,
  title={Upper Bounding Barlow Twins: A Novel Filter for Multi-Relational Clustering},
  author={Qian, Xiaowei and Li, Bingheng and Kang, Zhao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={13},
  pages={14660--14668},
  year={2024}
}
```
