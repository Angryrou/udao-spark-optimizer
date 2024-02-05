# udao-spark-optimizer


### Install

Using pip:

```bash
pip install -r requirements.txt
```

### Install on GPU

```bash
# first install autogluon on GPU
pip install -U pip
pip install -U setuptools wheel
# Install the proper version of PyTorch following https://pytorch.org/get-started/locally/
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install autogluon
pip install dglgo==0.0.2
pip install dgl>=1.1.2,<=1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html


# then install the rest of the dependencies
pip install -r requirements.txt
```
