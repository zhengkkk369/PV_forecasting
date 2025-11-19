## M
online_learning='full'
i=1
ns=(2 )
bszs=(1 )
lens=(1 24 48)
methods=('onenet_fsnet')
for n in ${ns[*]}; do
for bsz in ${bszs[*]}; do
for len in ${lens[*]}; do
for m in ${methods[*]}; do
CUDA_VISIBLE_DEVICES=0  nohup python -u main.py --method $m --root_path ./data/ --data_path PV1.csv --target Active_Power --n_inner $n --test_bsz $bsz --data custom --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 15 --learning_rate 1e-3 --online_learning $online_learning --use_adbfgs >  PV1$len$online_learning.out 2>&1 & 
# CUDA_VISIBLE_DEVICES=0  nohup python -u main.py --method $m --root_path ./data/ --data_path PV1.csv --features MS --c_out 1 --target Active_Power --n_inner $n --test_bsz $bsz --data custom --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 15 --learning_rate 1e-3 --online_learning $online_learning --use_adbfgs >  PV1$len$online_learning.out 2>&1 & 
done
done
done
done










