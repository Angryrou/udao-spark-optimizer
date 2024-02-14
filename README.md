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

## Additional dependencies for compile-time optimization (CPU/GPU/MacOS-intel)

```bash
pip install scikit-learn-intelex # only for intel chips
```

#### Possible issue on GPU

If there is an error related to "GLIBCXX_3.4.29" occuring, please try the following to export the "LD_LIBRARY_PATH":

```bash
find /your-home-directory -name "libstdc++.so*"
strings /your-directory-to-envs/your-env-name/lib/libstdc++.so.6.0.32 | grep GLIBCXX_3.4.29
export LD_LIBRARY_PATH=/your-directory-to-envs/your-env-name/lib/:$LD_LIBRARY_PATH
```
