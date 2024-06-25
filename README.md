# Multistage Net

Multistage Net: Learning Continuous Multistage Manufacturing Processes of Liquid Products without Intermediate Output and Lead-time Labels

Abstract: Manufacturers have implemented continuous multistage manufacturing processes (MMPs) for their efficiency and flexibility, especially in high-volume production of liquid products. While significant attention has been given to the development of data-driven soft sensors in manufacturing fields, research explicitly addressing continuous MMPs of liquid products is very scarce due to the following challenges: obtaining intermediate output labels and determining lead-time between stages. To overcome these challenges, we introduce Multistage Net, a novel machine learning model designed for continuous MMPs of liquid products. In Multistage Net, several interstage blocks are organized in a hierarchical structure within a multistage module. The interstage block is proposed to capture the sequential dependency between the previous and current stages and concurrently explores the lead-time relationship. From the interconnected interstage blocks, the multistage module can learn the sequential nature of MMPs across all stages even in the absence of intermediate output labels. Through validation experiments on two real-world datasets, it is shown that Multistage Net demonstrates superior prediction performance compared to baseline models. Moreover, further analysis reveals that the prediction performance of Multistage Net is not significantly impacted by the non-existence of lead-time labels.

## Get Started
1. Install Python 3.7, Pytorch 1.9.0
2. Train the model. We provide experiment scripts under the foler scripts. 

```bash
bash ./scripts/run_seqlen_12.sh
bash ./scripts/run_seqlen_24.sh
bash ./scripts/run_seqlen_48.sh
```
