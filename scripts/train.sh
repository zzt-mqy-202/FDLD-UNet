cd $(dirname $0)/..

INPUT=./data/isic2017
OUTPUT=./checkpoints/isic2017

MODEL=fdld_unet

batch_size=1

epochs=20
min_epochs=0
valid_interval=10
test_interval=50

opt=RMSprop
lr=1e-4
momentum=0.9
weight_decay=1e-4

sched=cosine

python train.py \
    --input $INPUT --output $OUTPUT \
    --model $MODEL \
    --batch_size $batch_size \
    --epochs $epochs --min_epochs $min_epochs\
    --valid_interval $valid_interval --test_interval $test_interval\
    --opt $opt --lr $lr --momentum $momentum --weight_decay $weight_decay \
    --sched $sched
