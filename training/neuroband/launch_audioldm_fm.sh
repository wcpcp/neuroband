CUDA_VISIBLE_DEVICES=1,2,3,4 python main.py --base ./configs/ldm_training/audioldm_large_cfm.yaml -t --resume /workspace/data_dir/tmp_dyh/Diff-Foley/training/neuroband/logs/2024-11-29T08-02-45_neuroband/checkpoints/epoch=000107.ckpt  --gpus 0,1,2,3 --stage 2 --epoch 250 --wandb_project diff_foley --scale_lr False