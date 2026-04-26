rm -rf results/res_no_aux
rm -rf results/res_pred_aux
rm -rf results/res_val_aux
rm -rf results/res_both_aux

python3 trainer.py --config configs/config_no_aux.txt
python3 trainer.py --config configs/config_pred_aux.txt
python3 trainer.py --config configs/config_val_aux.txt
python3 trainer.py --config configs/config_both_aux.txt