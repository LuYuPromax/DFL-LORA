## ConLoRA
ConLoRA is a platform for distributed fine-tuning of large models in a fully decentralized federated learning scenario. Relying on this platform, you can perform fine-tuning of large models in a completely decentralized network. In particular, to address the inherent error issues in decentralized federated learning, this platform allows you to observe the consensus error amplification effect caused by LoRA and provides the option to mitigate this error by freezing the A matrix during training.

## How to use
### Download
Go to the local folder where you want to store this platform and execute the following command
```
git clone https://github.com/LuYuPromax/ConLoRA.git
```

### Create environment
```
conda create -n env_name python==3.10.10
```
Then install related libraries
```
pip install -r requirement.txt
```

### Datasets
Currently supports some GLUE datasets and GSM8k datasets.First, go to the utils folder, modify the corresponding paths in glue_split.sh and gsm8k_split.sh to the local dataset storage path, and then execute the following instructions:
##### GLUE
```
chmod +x glue_split.sh
./glue_split.sh
```
##### GSM8K
```
chmod +x gsm8k_split.sh
./gsm8k_split.sh
```
Similarly, by running view_split.sh, you can view the data distribution of different clients.

### Finetune
