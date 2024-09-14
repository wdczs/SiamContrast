export CUDA_VISIBLE_DEVICES=0

python multitask_train.py --config=cfgs/hiucdv2_multi_bs8_siamcontrast.yaml
python test.py --config=cfgs/hiucdv2_multi_bs8_siamcontrast.yaml