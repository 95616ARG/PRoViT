for seed in 0; do
for method in provitLP stdLP; do
for n in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40; do
for netname in vitb16; do
for metric in 2; do
    mkdir -p experiments/logs/time_comp/${method}
    log="logs/time_comp/${method}/${netname}_lastlayer_n=${n}_metric=${metric}_seed=${seed}.log"
    python3 experiments/vit_repair.py \
        --path /home/public/datasets/ImageNet/ \
        --netname ${netname} \
        --device cuda:0 \
        --n ${n} \
        --seed ${seed} \
        --metric ${metric} \
        --method ${method} \
        --ft_niter 1 \
        --batch_size 100 \
    | tee -a ${log}
done
done
done
done
done
