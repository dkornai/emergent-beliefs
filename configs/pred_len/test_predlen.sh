rm -rf results/res_pred1_aux
rm -rf results/res_pred2_aux
rm -rf results/res_pred3_aux


python3 trainer.py --config configs/predlen/pred1_aux_config.txt
python3 trainer.py --config configs/predlen/pred2_aux_config.txt
python3 trainer.py --config configs/predlen/pred3_aux_config.txt