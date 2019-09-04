#!/bin/bash/
# For Testing
# 3x 
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --save 'DRLN_BD_Set5' --testpath ../LR/LRBD --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_BD_Set5' --testpath ../LR/LRBD --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --save 'DRLN_BD_Set14' --testpath ../LR/LRBD --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_BD_Set14' --testpath ../LR/LRBD --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --save 'DRLN_BD_B100' --testpath ../LR/LRBD --testset B100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_BD_B100' --testpath ../LR/LRBD --testset B100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --save 'DRLN_BD_Urban100' --testpath ../LR/LRBD --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_BD_Urban100' --testpath ../LR/LRBD --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --save 'DRLN_BD_Manga109' --testpath ../LR/LRBD --testset Manga109

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_BD_Manga109' --testpath ../LR/LRBD --testset Manga109

