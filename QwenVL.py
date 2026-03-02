from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import  PeftModel
from qwen_vl_utils import process_vision_info
import torch.nn as nn

class QwenVL(nn.Module):
    def __init__(self, cfg: dict):
        model_dir = cfg['llm_path']
        adapter_ckpt = cfg['adapter_ckpt']
        d_model = cfg['d_model']
        self.qwenvl = model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, device_map="auto",)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = PeftModel.from_pretrained(self.qwenvl, adapter_ckpt)
        vlm_hidden_size = self.qwenvl.config.cross_attention_hidden_size
        self.out = nn.Linear(vlm_hidden_size, d_model)


    def forward(self, messages ):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True,
                                                                       image_patch_size=16,
                                                                       return_video_metadata=True)
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas,
                           **video_kwargs, do_resize=False, return_tensors="pt")
        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.out(generated_ids)