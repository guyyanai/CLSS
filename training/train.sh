#!/bin/bash

python "train.py" \
--batch-size 180 \
--learning-rate 0.001 \
--projection-dim 32 \
--dataset-path "/users/kolodny/gyanai/CLSS/datasets/ecod-af2.csv" \
--structures-dir "/local/ecod/AF2/structures" \
--train-pickle "/users/kolodny/gyanai/CLSS/pickles/train-100.pkl" \
--validation-pickle "/users/kolodny/gyanai/CLSS/pickles/validation-100.pkl" \
--structures-dir "/local/ecod/AF2/structures" \
--checkpoint-path "/users/kolodny/gyanai/CLSS/checkpoints" \
--dataset-limit 100 \
--epochs 2 \
--random-sequence-stretches \
--random-stretch-min-size 10 \
--run-name "test-test"