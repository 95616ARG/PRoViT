for seed in 0; do
for method in provitLP provitFT provitFTLP; do
for n in 100 200 300 400; do
for netname in vitl32 deit; do
for metric in 4; do
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