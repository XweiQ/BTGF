# BTGF
This is code for [*Upper Bounding Barlow Twins: A Novel Filter for Multi-relational Clustering*](https://arxiv.org/abs/2312.14066) AAAI-24.

Overall node clustering result:

![image-20231228103143576](https://s2.loli.net/2023/12/28/aQPhOAjN6sxSb74.png)

The statistics of the datasets are as follows:

<img src="https://s2.loli.net/2023/12/28/n4myzZlYp7ONet5.png" alt="image-20231228103516094" style="zoom:80%;" />

The `learning rate` and `weight decay` of the optimizer are set to $1e^{−2}$ and $1e^{−3}$. 

The filter’s parameters $k$ and $γ$ is tuned in $[1, 2, 3, 4]$ and $[1, 10, 100, 1000]$, respectively.

The filter’s parameters $k$ and $γ$ are tuned in $[1, 2, 3, 4]$ and $[1, 1, 10, 100, 1000]$, respectively.

You can run BTGF with commands in the `script.sh` (e.g.: ACM)

```shell
python main.py -dataset ACM -epoch 400 -lr 1e-2 -wd 1e-3 -k 4 -a 10

python main.py -dataset amazon -epoch 400 -lr 1e-2 -wd 1e-3 -k 2 -a 1

python main.py -dataset aminer -epoch 400 -lr 1e-2 -wd 1e-3 -k 3 -a 100

python main.py -dataset DBLP_L -epoch 400 -lr 1e-2 -wd 1e-3 -k 2 -a 1000
```

