#!/bin/bash

#0 config
model_dir=/locate_data/zzh/models/
data_dir=/locate_data/public_data/criteo_test
batch_size=2048
num_epochs=1
max_steps=1000
task_type=train

case $1 in
    fm)
        model="FM"
    ;;
    dfm)
        model="DeepFM"
    ;;
esac

if [ -n "$2" ];then

task_type=$2
#echo ${task_type}
fi

#exit

#1 feature pipline
#python Feature_pipeline/get_criteo_feature.py --input_dir=${data_dir}/ --output_dir=${data_dir}/ --cutoff=200

#2 model pipline
#python Model_pipeline/wide_n_deep.py --model_type=wide --num_epochs=1 --batch_size=128 --deep_layers=256,128,64 --log_steps=1000 --num_threads=8 --model_dir=${model_dir}/lr/ --data_dir=${data_dir}
#python Model_pipeline/wide_n_deep.py --model_type=wide_n_deep --num_epochs=1 --batch_size=128 --deep_layers=256,128,64 --log_steps=1000 --num_threads=8 --model_dir=${model_dir}/wide_n_deep/ --data_dir=${data_dir}
python3 Model_pipeline/${model}.py --learning_rate=0.0001 \
    --max_steps=${max_steps} \
    --optimizer=Adam \
    --num_epochs=1 --embedding_size=32 \
    --batch_size=${batch_size} \
    --field_size=39 \
    --feature_size=117581 \
    --deep_layers=256,128 \
    --dropout=0.8,0.8 \
    --l2_reg=0.0001 \
    --log_steps=1000 \
    --num_threads=8 \
    task_type=${task_type} \
    --model_dir=${model_dir}/${model}/ --data_dir=${data_dir}

#python Model_pipeline/FM.py --learning_rate=0.0001 --optimizer=Adam --num_epochs=1 --embedding_size=32 --batch_size=256 --field_size=39 --feature_size=117581 --deep_layers=256,128 --dropout=0.8,0.8 --l2_reg=0.0001 --log_steps=1000 --num_threads=8 --model_dir=${model_dir}/FM/ --data_dir=${data_dir}
#python PNN.py --model_type=FNN --learning_rate=0.0001 --optimizer=Adam --num_epochs=1 --embedding_size=32 --batch_size=2048 --field_size=39 --feature_size=117581 --deep_layers=256,128 --dropout=0.8,0.8 --l2_reg=0.0001 --log_steps=1000 --num_threads=8 --model_dir=${model_dir}/FNN/ --data_dir=${data_dir}
#python PNN.py --model_type=Inner --learning_rate=0.0001 --optimizer=Adam --num_epochs=1 --embedding_size=32 --batch_size=2048 --field_size=39 --feature_size=117581 --deep_layers=256,128 --dropout=0.8,0.8 --l2_reg=0.0001 --log_steps=1000 --num_threads=8 --model_dir=${model_dir}/FNN/ --data_dir=${data_dir}
#python PNN.py --model_type=Outer --learning_rate=0.0001 --optimizer=Adam --num_epochs=1 --embedding_size=32 --batch_size=2048 --field_size=39 --feature_size=117581 --deep_layers=256,128 --dropout=0.8,0.8 --l2_reg=0.0001 --log_steps=1000 --num_threads=8 --model_dir=${model_dir}/FNN/ --data_dir=${data_dir}
#python NFM.py --learning_rate=0.00005 --optimizer=Adam --num_epochs=1 --batch_size=2048 --embedding_size=256 --deep_layers=256,128 --dropout=0.5,0.5,0.5 --l2_reg=0.001 --field_size=39 --feature_size=117581 --log_steps=1000 --num_threads=8 --model_dir=${model_dir}/NFM/ --data_dir=${data_dir} --batch_norm=True
#python AFM.py --learning_rate=0.0005 --optimizer=Adam --num_epochs=1 --batch_size=2048 --embedding_size=256 --attention_layers=128 --dropout=0.5,0.5 --l2_reg=0.001 --field_size=39 --feature_size=117581 --log_steps=1000 --num_threads=8 --model_dir=${model_dir}/AFM/ --data_dir=${data_dir}
#python DCN.py --learning_rate=0.0001 --optimizer=Adam --num_epochs=1 --embedding_size=32 --batch_size=2048 --field_size=39 --feature_size=117581 --deep_layers=512,256 --cross_layers=3 --dropout=0.8,0.8 --l2_reg=0.00001 --log_steps=1000 --num_threads=8 --model_dir=${model_dir}/DCN/ --data_dir=${data_dir}
#python DCN.py --dist_mode=2 --learning_rate=0.0001 --optimizer=Adam --num_epochs=1 --embedding_size=32 --batch_size=2048 --field_size=39 --feature_size=117581 --deep_layers=256,128 --cross_layers=2 --dropout=0.8,0.8 --l2_reg=0.0001 --log_steps=1000 --data_dir=/data/ceph/deep_ctr/data/criteo/ --model_dir=/data/ceph/deep_ctr/model_ckpt/ --clear_existing_model=True
#python DeepMVM.py --learning_rate=0.0001 --optimizer=Adam --num_epochs=1 --embedding_size=32 --batch_size=2048 --field_size=39 --feature_size=117581 --deep_layers=256,128 --dropout=0.8,0.8 --l2_reg=0.0001 --log_steps=1000 --num_threads=8 --model_dir=${model_dir}/DCN/ --data_dir=${data_dir}

#3 serving pipline
#python DeepFM.py --task_type=export --learning_rate=0.0005 --optimizer=Adam --batch_size=2048 --field_size=39 --feature_size=117581 --deep_layers=400,400,400 --dropout=0.5,0.5,0.5 --log_steps=1000 --num_threads=8 --model_dir=./model_ckpt/criteo/DeepFM/ --servable_model_dir=./servable_model/
