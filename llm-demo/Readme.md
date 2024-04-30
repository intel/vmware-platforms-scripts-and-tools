# Running a chatbot interface using Gradio* and Intel® Extension for PyTorch*
## Setup
The following sample is based on Intel® Extension for PyTorch* version 2.1.0. Please follow https://github.com/intel/intel-extension-for-pytorch/tree/v2.1.100%2Bcpu/examples/cpu/inference/python/llm to setup the environment.

A `requirements.txt` is attached to install the additional dependencies in the environment. This can be used by running `pip install -r requirements.txt`.

### Quantizing the model
Please follow the steps at https://github.com/intel/intel-extension-for-pytorch/tree/v2.1.100%2Bcpu/examples/cpu/inference/python/llm to quantize the model if necessary. To load a fine-tuned model with PEFT, please add the following lines to the model load: 

```
user_model = PeftModel.from_pretrained(user_model, '[PATH_TO_PEFT_WEIGHTS]')
user_model = user_model.merge_and_unload()
```

This script supports INT8 quantized models if provided

## Running the script
Run the script as follows: `python infer_server.py [ARGS]`

The following arguments are available to customize the script
- `-n/--name_or_path`: Path to the model files (The base model downloaded from huggingface). This is required regardless of the precision used since the files have the tokenizer details
- `-q/--quant_name_or_path`: Path to the quantized files. Use this if you are using a INT8 model. This should point to the output of the quantization
- `-p/--peft_path`: Path to the PEFT files. If you are using a fine-tuned model without quantization, you include the PEFT weight path

Additional CLI parameters can be looked at by using the `--help` option. By default, the inference runs on port 65535. For better performance, pinning is recommended by using numactl and VM pinning

## Disclaimer
When using any model with this script, you will need to agree to the applicable license terms for that particular model. This script has only been validated with the Llama-2 models. The model details and weights are available at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf. For other models, change to the input and output prompt format processing may be required



