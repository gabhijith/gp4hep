#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/cms/base/Miniconda/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cms/base/Miniconda/miniconda/etc/profile.d/conda.sh" ]; then
        . "/cms/base/Miniconda/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/cms/base/Miniconda/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

python run_limits_new.py --toyn=$1 --input_file=$2 --length_scale=$3 --variance=$4  --mean=$5 --sigma=$6 --rate_uc=$7 --mean_err=$8 --sigma_err=$9 --nwalkers=${10} --steps=${11} --sig_strength=${12} --show_result=False --length_scale_err=${13} --variance_err=${14}