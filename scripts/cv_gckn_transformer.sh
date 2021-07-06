cd ../experiments
cmd="python run_transformer_gckn_cv.py"
cmd_test="python run_transformer_gckn_cv.py --test"
dataset='MUTAG'
outdir="../logs_cv_gckn_trans/"
epochs=300
pos_enc="pstep"
#pos_enc="diffusion"
normalization="sym"
p=1
betas="1.0"
gckn_dims="32"
gckn_paths="5"
gckn_sigmas="0.6"
gckn_pooling="sum"
wd_list="0.01 0.001 0.0001"
nb_heads_list="1 4 8"
nb_layers_list="1 2 3"
dim_hidden_list="32 64 128"
lrs="0.001"
dropout_list="0.0"
folds="1 2 3 4 5 6 7 8 9 10"


startjob () {
    ${cmd} $1; ${cmd_test} $1
}

for gckn_path in $gckn_paths; do
for gckn_dim in $gckn_dims; do
for gckn_sigma in $gckn_sigmas; do
for nb_heads in $nb_heads_list; do
for wd in $wd_list; do
    for nb_layers in $nb_layers_list; do
        for dim_hidden in $dim_hidden_list; do
            for lr in $lrs; do
                for dropout in $dropout_list; do
                    for beta in $betas; do
                        for fold in $folds; do
                                startjob "--fold-idx ${fold} --weight-decay ${wd} --gckn-sigma ${gckn_sigma} --gckn-path ${gckn_path} --gckn-dim ${gckn_dim} --gckn-pooling ${gckn_pooling} --dataset ${dataset} --outdir ${outdir} --dropout ${dropout} --epochs ${epochs} --nb-heads ${nb_heads} --nb-layers ${nb_layers} --dim-hidden ${dim_hidden} --lr ${lr} --pos-enc ${pos_enc} --p ${p} --beta ${beta} --normalization ${normalization}"
                        done
                    done
                done
            done
        done
    done
done
done
done
done
done
