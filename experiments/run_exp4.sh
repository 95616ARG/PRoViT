for seed in 0; do
for method in provitLP provitFT provitFTLP stdLP; do
for n in 4 8 12 16 20; do
for netname in vitl32 deit; do
for metric in 1; do
    mkdir -p experiments/logs/time_comp/${method}
    log="logs/time_comp/${method}/${netname}_lastlayer_n=${n}_metric=${metric}_seed=${seed}.log"
    python3 experiments/small_vit.py \
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