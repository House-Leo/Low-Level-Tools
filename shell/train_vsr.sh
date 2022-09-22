# BasicVSRPP
# CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 --master_port=8975 basicsr/train.py -opt options/train/BasicVSRPP/train_BasicVSRPP_Vimeo90K_BI.yml --launcher pytorch

# CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 --master_port=8975 basicsr/train.py -opt options/train/BasicVSRPP/train_BasicVSRPP_REDS.yml --launcher pytorch

# BasicVSR
CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 --master_port=8975 basicsr/train.py -opt options/train/BasicVSR/train_BasicVSR_Vimeo90K_BIx4.yml --launcher pytorch