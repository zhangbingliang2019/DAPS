#!/bin/bash

# Usage:
#     ./scripts/download_models.sh celeba ffhq

MODELS=("celeba" "ffhq" "lsun_churches" "lsun_bedrooms" "text2img" "cin" "semantic_synthesis" "semantic_synthesis256" "sr_bsr" "layout2img_model" "inpainting_big")
DOWNLOAD_PATH="https://ommer-lab.com/files/latent-diffusion"

function download_models() {
    local list=("$@")

    for arg in "${list[@]}"; do
        for model in "${MODELS[@]}"; do
            if [[ "$model" == "$arg" ]]; then
                echo "Downloading $model"
                model_dir="./models/ldm/$arg"
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

download_models "$@"
