sampler:
	rlaunch --cpu 4 --charged-group is_jg_bokeh -P 1 --negative-tags=node/gpu-2080ti-0538.host.hh-d.brainpp.cn,node/gpu-2080ti-0262.host.hh-d.brainpp.cn,node/gpu-2080ti-0285.host.hh-d.brainpp.cn --comment='sampler_aug_big' --memory 164000 python sampler.py
preprocess:
	rlaunch --cpu 16  --gpu 8 --charged-group is_jg_bokeh -P 1 --memory 32000 --negative-tags=node/gpu-2080ti-0538.host.hh-d.brainpp.cn,node/gpu-2080ti-0262.host.hh-d.brainpp.cn,node/gpu-2080ti-0285.host.hh-d.brainpp.cn --comment='preprocess_aug_big' --preemptible no python preprocess.py
dp_split:
	rlaunch --cpu 8 --gpu 0 --charged-group is_jg_mp -P 16 --memory 64000 --comment='dp_aug_big' --negative-tags=node/gpu-2080ti-0538.host.hh-d.brainpp.cn,node/gpu-2080ti-0262.host.hh-d.brainpp.cn,node/gpu-2080ti-0285.host.hh-d.brainpp.cn --preemptible no python dp_split.py

train_a100:
	rlaunch --cpu 4 --memory 160000 --gpu 4 --comment='train_aug_big' \
    --charged-group=jg_renxiang_a100 \
    --private-machine=yes --positive-tags=A100 \
	--negative-tags=node/gpu-2080ti-0538.host.hh-d.brainpp.cn,node/gpu-2080ti-0262.host.hh-d.brainpp.cn,node/gpu-2080ti-0285.host.hh-d.brainpp.cn \
    -- python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=34328 train.py

train:
	rlaunch --cpu 8 --memory 160000 --gpu 8 --comment='train_aug_big' \
    --charged-group is_jg_bokeh \
    --negative-tags=node/gpu-2080ti-0538.host.hh-d.brainpp.cn,node/gpu-2080ti-0262.host.hh-d.brainpp.cn,node/gpu-2080ti-0285.host.hh-d.brainpp.cn \
    -- python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=42328 train.py

