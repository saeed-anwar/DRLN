#!/bin/bash/
# For Testing
# 2x
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --save 'DRLN_Set5' --testpath ../LR/LRBI --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Set5' --testpath ../LR/LRBI --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --save 'DRLN_Set14' --testpath ../LR/LRBI --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Set14' --testpath ../LR/LRBI --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --save 'DRLN_B100' --testpath ../LR/LRBI --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_B100' --testpath ../LR/LRBI --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --save 'DRLN_Urban100' --testpath ../LR/LRBI --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Urban100' --testpath ../LR/LRBI --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --save 'DRLN_Manga109' --testpath ../LR/LRBI --testset Manga109

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Manga109' --testpath ../LR/LRBI --testset Manga109


# 3x
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --save 'DRLN_Set5' --testpath ../LR/LRBI --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Set5' --testpath ../LR/LRBI --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --save 'DRLN_Set14' --testpath ../LR/LRBI --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Set14' --testpath ../LR/LRBI --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --save 'DRLN_B100' --testpath ../LR/LRBI --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_B100' --testpath ../LR/LRBI --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --save 'DRLN_Urban100' --testpath ../LR/LRBI --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Urban100' --testpath ../LR/LRBI --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --save 'DRLN_Manga109' --testpath ../LR/LRBI --testset Manga109

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Manga109' --testpath ../LR/LRBI --testset Manga109

# 4x
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX4/DRLN_BIX4.pt --test_only --save_results --chop --save 'DRLN_Set5' --testpath ../LR/LRBI --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX4/DRLN_BIX4.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Set5' --testpath ../LR/LRBI --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX4/DRLN_BIX4.pt --test_only --save_results --chop --save 'DRLN_Set14' --testpath ../LR/LRBI --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX4/DRLN_BIX4.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Set14' --testpath ../LR/LRBI --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX4/DRLN_BIX4.pt --test_only --save_results --chop --save 'DRLN_B100' --testpath ../LR/LRBI --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX4/DRLN_BIX4.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_B100' --testpath ../LR/LRBI --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX4/DRLN_BIX4.pt --test_only --save_results --chop --save 'DRLN_Urban100' --testpath ../LR/LRBI --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX4/DRLN_BIX4.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Urban100' --testpath ../LR/LRBI --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX4/DRLN_BIX4.pt --test_only --save_results --chop --save 'DRLN_Manga109' --testpath ../LR/LRBI --testset Manga109

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX4/DRLN_BIX4.pt --test_only --save_results --chop --self_ensemble --save 'DRLNplus_Manga109' --testpath ../LR/LRBI --testset Manga109

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

