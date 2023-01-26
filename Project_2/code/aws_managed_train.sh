#!/bin/bash

# Setup environment (do not change this)
source activate pytorch_p36
pip install -r requirements.txt
export WANDB_API_KEY=$(cat "aws_configs/wandb.key")

# Download dataset (do not change this)
if [ ! -d "/home/ubuntu/miniscapes" ]; then
  echo "Download miniscapes"
  aws s3 cp s3://dlad-miniscapes-2021/miniscapes.zip /home/ubuntu/
  echo "Extract miniscapes"
  unzip /home/ubuntu/miniscapes.zip -d /home/ubuntu/ | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
  rm /home/ubuntu/miniscapes.zip
  echo "\n"
fi

# Run training (do not change this)
echo "Start training"
cd /home/ubuntu/code/

# Please change the hyperparameters in mtl/scripts/managed_spot_train.
python -m mtl.scripts.managed_spot_train

# Wait a moment before stopping the instance to give a chance to debug
echo "Terminate instance in 2 minutes. Use Ctrl+C to cancel the termination. After using Ctrl+C you are responsible for stopping the instance yourself."
sleep 2m && bash aws_stop_self.sh
