#!/bin/bash/
# For Testing
# 8x
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX8/DRLN_BIX8.pt --test_only --save_results --chop --save 'DRLN_Set5' --testpath ../LR/LRBI --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX8/DRLN_BIX8.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Set5' --testpath ../LR/LRBI --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX8/DRLN_BIX8.pt --test_only --save_results --chop --save 'DRLN_Set14' --testpath ../LR/LRBI --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX8/DRLN_BIX8.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Set14' --testpath ../LR/LRBI --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX8/DRLN_BIX8.pt --test_only --save_results --chop --save 'DRLN_B100' --testpath ../LR/LRBI --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX8/DRLN_BIX8.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_B100' --testpath ../LR/LRBI --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX8/DRLN_BIX8.pt --test_only --save_results --chop --save 'DRLN_Urban100' --testpath ../LR/LRBI --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX8/DRLN_BIX8.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Urban100' --testpath ../LR/LRBI --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX8/DRLN_BIX8.pt --test_only --save_results --chop --save 'DRLN_Manga109' --testpath ../LR/LRBI --testset Manga109

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX8/DRLN_BIX8.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Manga109' --testpath ../LR/LRBI --testset Manga109
