import time
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "ByteDance/Dolphin"
image_path = "./images/2401_1.png"

processor = DonutProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device)

image = Image.open(image_path)
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device).to(torch.float16)
print(f"pixel_values size: {pixel_values.size()}")

tokenizer = processor.tokenizer

prompt = "<s>Parse the reading order of this document. <Answer/>"
data = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
prompt_ids = data["input_ids"].to(device)
decoder_attention_mask = data["attention_mask"].to(device)
# decoder_attention_mask = torch.ones_like(prompt_ids)
print(f"prompt_ids: {prompt_ids}, size: {prompt_ids.size()}")

start_time = time.time()
outputs = model.generate(
    pixel_values=pixel_values,
    decoder_input_ids=prompt_ids,
    decoder_attention_mask=decoder_attention_mask,
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[tokenizer.unk_token_id]],
    return_dict_in_generate=True,
    do_sample=False,
    num_beams=1,
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"cost: {elapsed_time * 1000:.2f} ms")

print(f"output_ids: {outputs.sequences}, size: {outputs.sequences.size()}")
sequence = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
sequence = sequence.replace(prompt, "").replace("</s>", "").replace("<pad>", "").strip()
print(sequence)
