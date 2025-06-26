import vllm_dolphin
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt
from transformers import DonutProcessor

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "ByteDance/Dolphin"
image_path = "./images/2401_1.png"
image = Image.open(image_path)

##########################################################################################
processor = DonutProcessor.from_pretrained(model_id)

encoder_prompt = ["0"] * 783
encoder_prompt = "".join(encoder_prompt)
# encoder_prompt_tokens = TokensPrompt(
#     prompt_token_ids=processor.tokenizer(encoder_prompt, add_special_tokens=False)["input_ids"]
# )

decoder_prompt = "<s>Parse the reading order of this document. <Answer/>"
decoder_prompt_tokens = TokensPrompt(
    prompt_token_ids=processor.tokenizer(decoder_prompt, add_special_tokens=False)["input_ids"]
)
enc_dec_prompt = ExplicitEncoderDecoderPrompt(
    encoder_prompt=TextPrompt(prompt=encoder_prompt, multi_modal_data={"image": image}),
    decoder_prompt=decoder_prompt_tokens,
)

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    logprobs=0,
    prompt_logprobs=None,
    skip_special_tokens=False,
)

# Create an encoder/decoder model instance
llm = LLM(
    model=model_id,
    dtype="float32",    # if cuda is available, use "float16"
    enforce_eager=True,
    trust_remote_code=True,
    max_num_seqs=8,
    hf_overrides={"architectures": ["DolphinForConditionalGeneration"]},
)

# Generate output tokens from the prompts. The output is a list of
# RequestOutput objects that contain the prompt, generated text, and other information.
outputs = llm.generate(
    prompts=[enc_dec_prompt, enc_dec_prompt],
    sampling_params=sampling_params,
    use_tqdm=True,
)

print("------" * 8)

# Print the outputs.
for output in outputs:
    decoder_prompt_tokens = processor.tokenizer.batch_decode(output.prompt_token_ids, skip_special_tokens=True)
    decoder_prompt = "".join(decoder_prompt_tokens)
    generated_text = output.outputs[0].text
    print(f"Decoder prompt: {decoder_prompt!r}, "
          f"\nGenerated text: {generated_text!r}")

    print("------" * 8)

# ------------------------------------------------
# Decoder prompt: 'Parse the reading order of this document.',
# Generated text: ' [0.25,0.12,0.75,0.15] title[PAIR_SEP][0.32,0.20,0.68,0.21] author[PAIR_SEP][0.38,0.21,0.62,0.24] para[PAIR_SEP][0.34,0.24,0.65,0.27] para[PAIR_SEP][0.38,0.31,0.62,0.51] fig[PAIR_SEP][0.47,0.52,0.53,0.53] sec[PAIR_SEP][0.29,0.55,0.71,0.67] para[PAIR_SEP][0.25,0.70,0.35,0.71] sec[PAIR_SEP][0.25,0.73,0.75,0.88] para[PAIR_SEP][0.27,0.89,0.45,0.90] fnote[PAIR_SEP][0.25,0.92,0.41,0.94] foot[PAIR_SEP][0.14,0.28,0.16,0.70] watermark'
# ------------------------------------------------
# Decoder prompt: 'Parse the reading order of this document.',
# Generated text: ' [0.25,0.12,0.75,0.15] title[PAIR_SEP][0.32,0.20,0.68,0.21] author[PAIR_SEP][0.38,0.21,0.62,0.24] para[PAIR_SEP][0.34,0.24,0.65,0.27] para[PAIR_SEP][0.38,0.31,0.62,0.51] fig[PAIR_SEP][0.47,0.52,0.53,0.53] sec[PAIR_SEP][0.29,0.55,0.71,0.67] para[PAIR_SEP][0.25,0.70,0.35,0.71] sec[PAIR_SEP][0.25,0.73,0.75,0.88] para[PAIR_SEP][0.27,0.89,0.45,0.90] fnote[PAIR_SEP][0.25,0.92,0.41,0.94] foot[PAIR_SEP][0.14,0.28,0.16,0.70] watermark'
# ------------------------------------------------