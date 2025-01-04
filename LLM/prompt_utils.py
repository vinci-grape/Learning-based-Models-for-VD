



def prompt_model_related_pre_processing(model_name: str, prompt: str):
    prompt = prompt.lstrip()
    if model_name == "WizardLM/WizardLM-7B-V1.0":
        return f"""{prompt}
### Response:"""
    elif model_name == 'mosaicml/mpt-7b-instruct':
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{prompt}
### Response:"""
    elif model_name == 'TheBloke/stable-vicuna-13B-HF':  # vicuna 1.0 version
        return f"""### Human:{prompt}
### Assistant:"""
    elif model_name in ['TheBloke/wizard-vicuna-13B-HF', 'eachadea/vicuna-13b-1.1']:  # vicuna 1.1 version
        return f"""USER: {prompt}
ASSISTANT:"""
    elif model_name == 'TheBloke/koala-13B-HF':
        return f"""### Instruction:{prompt}
### Response:"""
    elif model_name == 'tiiuae/falcon-40b-instruct':
        return f""">>QUESTION<<{prompt}
>>ANSWER<<"""
    elif model_name == 'project-baize/baize-v2-13b':
        return f"""[|Human|]{prompt}
[|AI|]"""
    return prompt


def decode_str_post_processing(model_name: str, prompt: str, decode_str: str):
    """
        decode_str = prompt + generate_str
    """
    generate_str = decode_str[len(prompt):]
    if model_name == 'project-baize/baize-v2-13b':
        stop_words = ['[|AI|]', '[|Human|]']
        for stop_word in stop_words:
            if stop_word in generate_str:
                generate_str = generate_str[:generate_str.index(stop_word)]
        return f"{prompt}{generate_str}"

    return decode_str