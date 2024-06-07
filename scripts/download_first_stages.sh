#!/bin/bash

# Usage:
#     ./scripts/download_first_stages.sh kl-f4 kl-f8

MODELS=("kl-f4" "kl-f8" "kl-f16" "kl-f32" "vq-f4" "vq-f4-noattn" "vq-f8" "vq-f8-n256" "vq-f16")
DOWNLOAD_PATH="https://ommer-lab.com/files/latent-diffusion"

function download_first_stages() {
    local list=("$@")

    for arg in "${list[@]}"; do
        for model in "${MODELS[@]}"; do
            if [[ "$model" == "$arg" ]]; then
                echo "Downloading $model"
                model_dir="./models/first_stage_models/$arg"
                if [ ! -d "$model_dir" ]; then
                    mkdir -p "$model_dir"
                    echo "Directory created: $model_dir"
                else
                    echo "Directory already exists: $model_dir"
                fi
                wget -O "$model_dir/model.zip" "$DOWNLOAD_PATH/$arg.zip"
                unzip -o "$model_dir/model.zip" -d "$model_dir"
                rm -rf "$model_dir/model.zip"
            fi
        done
    done
}

download_first_stages "$@"
