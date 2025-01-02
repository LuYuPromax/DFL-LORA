# ConLoRA
ConLoRA is a platform for distributed fine-tuning of large models in a fully decentralized federated learning scenario. Relying on this platform, you can perform fine-tuning of large models in a completely decentralized network. In particular, to address the inherent error issues in decentralized federated learning, this platform allows you to observe the consensus error amplification effect caused by LoRA and provides the option to mitigate this error by freezing the A matrix during training.

# Background
Large Foundation Models, such as Large Language Models (LLMs), Vision Transformers (ViTs), and multimodal models, exhibit the ability to be fine-tuned on task-specific datasets to produce customized solutions for numerous downstream applications. As the FMs grow increasingly complex (often exceeding 400 billion parameters), the fine-tuning process becomes computationally intensive. Parameter Efficient Fine-Tuning (PEFT), particularly LoRA, has emerged as a pivotal approach for model fine-tuning. LoRA enables the meticulous adjustment of a selected subset of model parameters, significantly reducing the computational cost of model fine-tuning.

Recently, Federated Learning (FL) has been increasingly integrated with LoRA. The architecture includes a central parameter server to synchronize the local parameters of the local low-rank matrices B and A of LoRA. The centralized synchronization enables parallel fine-tuning of local FMs across multiple devices while keeping data localized. However, the practicality of a centralized server for global synchronization diminishes with expanding network sizes and privacy concerns. In contrast, Decentralized FL (DFL) offers a practical alternative, facilitating peer-to-peer model synchronization among devices without a central coordinator. Owing to the features of DFL, the integration of DFL and LoRA is appealing for real-world data-sharing initiatives among various organizations, as it offers enhanced privacy, robustness, and scalability.

However, designing an efficient and robust framework that holistically integrates DFL with LoRA presents significant challenges. In the FL setting, the centralized parameter server can perform the global aggregation, ensuring precise synchronization across all local low-rank matrices. In DFL, however, the absence of a centralized server results in a peer-to-peer distributed consensus mechanism that is prone to consensus errors, leading to divergent local low-rank matrices B and A among clients. These consensus errors are exacerbated after the matrix multiplication in local FMs,resulting in the degradation of model accuracy.

# How to use
## Create a conda environment


