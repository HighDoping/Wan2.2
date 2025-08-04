# Wan2.2 Text-to-Video Model for Mac M-Series

**WORK IN PROGRESS**

This repository has the sole purpose of making Wan2.2 run efficiently on Mac M-Series chips. Memory saving is the priority.

## Original problems

Mac M-Series chips have Unified Memory, so the original method of offloading models to the CPU still costs memory.

The original repo also loads all models at startup, which takes a lot of memory. (umt5-xxl is a 13B model!!)

## Changes

- Load models only when needed. (T5, base model, and vae)
- Modify the offload_model method to delete the model from memory immediately after use.
- Add quantized T5 model to reduce memory usage.
- Use bf16 precision to reduce memory usage.

## Problems

- VAE tiling for VAE2.2 is broken now.

## TODO

- Fix VAE tiling for VAE2.2.
- Add support for A14B model.

## Installation

Follow the upstream instructions to install the dependencies and download the model.

Assuming you have Poetry installed, you can also install the dependencies with:

```bash
poetry install
```

If you want to use hugingface-cli or modelscope, you can install with:

```bash
poetry install --extras dev
```

Download the model with huggingface-cli or modelscope:

```bash
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

```bash
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
modelscope download Wan-AI/Wan2.2-I2V-A14B --local_dir ./Wan2.2-T2V-A14B
modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir ./Wan2.2-TI2V-5B
```

To use quantized T5 model, [download it](https://huggingface.co/HighDoping/umt5-xxl-encode-gguf/resolve/main/umt5-xxl-encode-only-Q4_K_M.gguf) from my [🤗 repo](https://huggingface.co/HighDoping/umt5-xxl-encode-gguf) or use huggingface-cli and put it in the same folder as wan model:

```bash
huggingface-cli download HighDoping/umt5-xxl-encode-gguf --local-dir ./Wan2.2-TI2V-5B
```

[Models from city96](https://huggingface.co/city96/umt5-xxl-encoder-gguf) also works. Only needs to change the model name in ```wan\configs```

Then install llama.cpp from homebrew:

```bash
brew install llama.cpp
```

## Usage

### Text to video with Wan2.2-TI2V-5B

To generate a video, use the following command:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task ti2v-5B --size "1280*704" --frame_num 41 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_quant --device mps --prompt "Penguins fighting a polar bear in the arctic." --save_file output_video.mp4
```

```--t5_quant``` enables the quantized T5 model.

For 32GB M4 Mac Mini, time taken: 1h37m. Result: [TI2V_T2V](./assets/TI2V_T2V.mp4)

### Image to video with Wan2.2-TI2V-5B

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py --task ti2v-5B --size "1280*704" --frame_num 25 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_quant --device mps --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

For 32GB M4 Mac Mini, time taken: 47m. Result: [TI2V_I2V](./assets/TI2V_I2V.mp4)

### How to choose the parameters

- **```--frame_num```**: The number of frames to generate. The default is 81. The output video is at 16 FPS, so 81 frames is 5 seconds. You should choose a number that is 4n+1. Generation time and memory usage is proportional to the number of frames. Too short video will not look good.

- **```--sample_steps```**: The number of steps to sample. The default is 50 for T2V and 40 for I2V. Generation time increase linearly to the number of steps. The more steps, the better the quality. But it also takes longer to generate.