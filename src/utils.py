
from transformers import AutoTokenizer, PreTrainedTokenizer
from trl import ModelConfig

from prompts import PROMPTS

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

PAD_TOKEN_MAP = {
    "DeepSeek-R1-Distill-Llama-8B": "<|finetune_right_pad_id|>",
    "DeepSeek-R1-Distill-Llama-70B": "<|finetune_right_pad_id|>",
    "DeepSeek-R1-Distill-Qwen-1.5B": "<|video_pad|>",
    "DeepSeek-R1-Distill-Qwen-7B": "<|video_pad|>",
    "DeepSeek-R1-Distill-Qwen-14B": "<|video_pad|>",
    "DeepSeek-R1-Distill-Qwen-32B": "<|video_pad|>",
    "Qwen2.5-0.5B": "<|video_pad|>",
    "Qwen2.5-1.5B": "<|video_pad|>",
    "Qwen2.5-3B": "<|video_pad|>",
    "Qwen2.5-7B": "<|video_pad|>",
    "Qwen2.5-14B": "<|video_pad|>",
    "Qwen2.5-32B": "<|video_pad|>",
    "Qwen2.5-Math-1.5B": "<|video_pad|>",
    "Qwen2.5-Math-7B": "<|video_pad|>",
}


def get_tokenizer(
    model_args: ModelConfig, chat_template: str | None = None, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if chat_template is not None:
        tokenizer.chat_template = chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    model_name = model_args.model_name_or_path.split("/")[-1]    
    if model_name in PAD_TOKEN_MAP:
        pad_token = PAD_TOKEN_MAP[model_name]
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

    assert tokenizer.pad_token is not None 
    assert tokenizer.pad_token != tokenizer.eos_token and tokenizer.pad_token_id != tokenizer.eos_token_id

    return tokenizer

def push_model_to_hub(model_name_or_path, model_name_hub):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    model.push_to_hub(model_name_hub)
    tokenizer.push_to_hub(model_name_hub)
    print(f"Model and tokenizer pushed to Hugging Face Hub at {model_name_hub}")

if __name__ == "__main__":

    model_dir = "/mnt/lustrenew/mllm_safety-shared/lisiheng/checkpoints"
    
    model_name = "Qwen3-1.7B-DeepMath-1024samples-3-accuracy-512-1024-8-1.0-50-grpo-0.0-0.20-0.20-1-true-1e-6-32-highest_k-8-16-16-split-1.0-ReplayGRPO"    
    model_name_hub = "Siheng99/Qwen3-1.7B-DeepMath-1024samples-RePO"
    
    model_name_or_path = f"{model_dir}/{model_name}"
    
    push_model_to_hub(model_name_or_path, model_name_hub)
    