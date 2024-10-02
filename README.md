## Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning

####  [[Website]](https://weirdlabuw.github.io/vpl/) [[Paper]](https://arxiv.org/) 

[Sriyash Poddar<sup>1</sup>](https://sriya.sh), [Yanming Wan<sup>1</sup>](https://yanmingwan.com/), [Hamish Ivison<sup>1</sup>](https://hamishivi.github.io/), [Abhishek Gupta<sup>1</sup>](https://homes.cs.washington.edu/~abhgupta), [Natasha Jaques<sup>1</sup>](https://natashajaques.ai)<br/>

<sup>1</sup>University of Washington

This repo is an implementation of the control experiments of VPL. VPL is a varitional framework for learning from human feedback (binary preference labels) i.e. inferring a novel user-specific latent and learning reward models and policies conditioned on this latent without additional user-specific data. This is used for quick adaptation to specific user preferences without retraining the entire model or ignoring underrepresented groups.

## Instructions


#### Setting up repo
```
git clone git@github.com:WEIRDLabUW/vpl.git
```

#### Install Dependencies
```
conda create -n vpl python=3.10
conda activate vpl
pip install -r requirements.txt
pip install -e dependencies/d4rl --no-deps
pip install -e dependencies/ravens
pip install -e .
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Install "mujoco-py" from here: https://github.com/vwxyzjn/free-mujoco-py 
```

## Reward Learning

To create preference dataset for VPL:
```
python -m pref_learn.create_dataset --num_query=<num of comparison pairs> --env=<env_name> --query_len=<query_length> --set_len=<annotation_batch_size>
```

To train VPL on a preference dataset, run the following commands:
```
python pref_learn/train.py \
    --comment=<project_name> \
    --env=<env_name> \
    --dataset_path=<dataset_path> \
    --model_type=<reward_model_class> \
    --logging.output_dir="logs" \
    --seed "seed" \
```


## Policy Learning
To train VPL policies:

```
python experiments/run_iql.py \
        --env_name=<env_name> \
        --seed=<seed> \
        --model_type=<reward_model_type> \
        --ckpt=<reward_model_checkpoint_dir> \
        --use_reward_model=True
```


## Active Learning

To evaluate the VPL policy using actively queried samples:

```
python experiments/eval.py \
   --env_name=<env_name> \
   --samples=<eval_episodes>
   --sampling_method="information_gain" \
   --preference_dataset_path=<path to preference dataset> \
   --reward_model_path=<reward_model_checkpoint_dir> \
   --policy_path=<policy_checkpoint_dir>   
```

#### All examples scripts can be found in the scripts directory. Run the script as follows:

```
bash run.sh VAE max 
```


## Acknowledgement

This repository uses the IQL implementation in jax from https://github.com/dibyaghosh/jaxrl_m.


## Bibtex
If you find this code useful, please cite:

```
@article{poddar2024vpl,
    author    = {Poddar, Sriyash and Wan, Yanming and Ivision, Hamish and Gupta, Abhishek and Jaques, Natasha},
    title     = {Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning},
    booktitle = {ArXiv Preprint},
    year      = {2024},
}
```
