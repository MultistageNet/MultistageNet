# Multistage Net

Multistage Net: Learning Continuous Multistage Manufacturing Processes of Liquid Products without Intermediate Output and Lead-time Labels

Manufacturers have implemented continuous multistage manufacturing processes (MMPs) for their efficiency and flexibility, especially in high-volume production of liquid products. While significant attention has been given to the development of data-driven soft sensors in manufacturing fields, research explicitly addressing continuous MMPs of liquid products is very scarce due to the following challenges: obtaining intermediate output labels and determining lead-time between stages. To overcome these challenges, we introduce Multistage Net, a novel machine learning model designed for continuous MMPs of liquid products. In Multistage Net, several interstage blocks are organized in a hierarchical structure within a multistage module. The interstage block is proposed to capture the sequential dependency between the previous and current stages and concurrently explores the lead-time relationship. From the interconnected interstage blocks, the multistage module can learn the sequential nature of MMPs across all stages even in the absence of intermediate output labels. Through validation experiments on two real-world datasets, it is shown that Multistage Net demonstrates superior prediction performance compared to baseline models. Moreover, further analysis reveals that the prediction performance of Multistage Net is not significantly impacted by the non-existence of lead-time labels.

## Multistage Net Architecture
<p align="center">
<img src=".\pic\MultistageNet.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Multistage Net consists of three main components: 1) interstage blocks, 2) multistage modules, and 3) temporal encoders
</p>

## Dataset Description
<p align="center">
<img src=".\pic\LiquidSugar.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Liquid sugar production process, which is comprised of three sequential stages: melter, two parallel saturators, and the final saturator.
</p>

- The liquid sugar production process involves three main stages that sequentially process sugar cane into liquid sugar. 
- The raw materials (i.e., sugar canes) are continuously fed into the melter. They are transformed into intermediate products and then continuously passed to the subsequent stage, namely the two parallel saturators, through pipes. This process repeats until the final stage, where liquid sugar is produced. 
- Each main stage is equipped with multiple sensors monitoring control parameters including temperature, pressure, flow rate, and additive input. 
- The dataset includes 22,638 observations, covering the period from May 2021 to April 2022, gathered at 15-minute intervals from 18 sensors during production.
- Manual measurements of the sugar cane’s chemical ingredients, taken whenever new raw materials were introduced, were also included in the dataset. These measurements were incorporated as the control parameters and the manufacturing process state vary with the raw material's chemical ingredients.


## Get Started
1. Install Python >= 3.9, Pytorch 1.12.1
2. Train the model. We provide experiment scripts under the foler scripts. (Reproduce experiment results by under scripts:)

```bash
# Main script
bash ./scripts/run.sh

# Reproduce results
bash ./scripts/run_reproduce_12.sh
bash ./scripts/run_reproduce_24.sh
bash ./scripts/run_reproduce_48.sh
```

### Simple workflow example
We make simple workflow at 'train_predict.ipynb'
