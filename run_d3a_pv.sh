## M
online_learning='full'
i=1
ns=(2 )
bszs=(1 )
lens=(48)
methods=('fsnet_d3a')
sleep_interval=16
sleep_epochs=20
for n in ${ns[*]}; do
for bsz in ${bszs[*]}; do
for len in ${lens[*]}; do
for m in ${methods[*]}; do
online_adjust=0.5
offline_adjust=0.5
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --method $m --root_path ./data/ --data_path PV1.csv --n_inner $n --test_bsz $bsz --data custom --target Active_Power --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 6 --learning_rate 1e-3 --online_learning $online_learning  --sleep_interval $sleep_interval --sleep_epochs $sleep_epochs --online_adjust $online_adjust --offline_adjust $offline_adjust > $m-PV1$len$online_learning-sleep$sleep_interval-epoch$sleep_epochs'_'online_adjust$online_adjust'_'offline_adjust$offline_adjust.out 2>&1 & 
online_adjust=2.0
offline_adjust=2.0
CUDA_VISIBLE_DEVICES=0  nohup python -u main.py --method $m --root_path ./data/ --data_path PV1.csv --n_inner $n --test_bsz $bsz --data custom --target Active_Power --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 6 --learning_rate 3e-3 --online_learning $online_learning  --sleep_interval $sleep_interval --sleep_epochs $sleep_epochs --online_adjust $online_adjust --offline_adjust $offline_adjust > $m-PV1$len$online_learning-sleep$sleep_interval-epoch$sleep_epochs'_'online_adjust$online_adjust'_'offline_adjust$offline_adjust.out 2>&1 & 
done
done
done
done










