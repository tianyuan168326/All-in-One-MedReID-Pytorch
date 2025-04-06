CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port=28349 test_extract_id.py --lr 5e-6 \
--epochs 20 --optim adamw --save_every_n_epochs 3 --wd 5e-2 --backbone clip_Tadapt --clip_grad -1 --online_compute_id  \
--batch_size 6 --text_use_method identity_CL --mimic_all --lora_rank 16 --margin -1 --ft_all --relation_type difference --start_epoch 0 \
--exp_dir ./ \
--resume_main MAMI_pretrained.pth
