import json
import os
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_huggingface(questions, prompt):
    pretrained = os.environ["PRETRAINED_MODEL_NAME_OR_PATH"]
    model_kwargs = parse_model_kwargs(os.environ.get("MODEL_KWARGS", ""))
    tokenizer_kwargs = parse_model_kwargs(os.environ.get("TOKENIZER_KWARGS", ""))
    model = AutoModelForCausalLM.from_pretrained(pretrained, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, **tokenizer_kwargs)
    outputs = []
    for question in questions:
        input_ = create_input(prompt, question)
        input_ids = tokenizer(input_)
        input_ids = torch.Tensor(input_ids.input_ids)[None].to(torch.int).to("cuda")
        answer_ids = model.generate(input_ids, max_length=2048)
        # Remove input prefix from output
        assert (answer_ids[:, :input_ids.shape[1]] == input_ids).all()
        answer_ids_ = answer_ids[:, input_ids.shape[1]:]
        answer = tokenizer.batch_decode(answer_ids_, skip_special_tokens=True)[0]
        outputs.append(answer)
    return outputs, outputs


def parse_model_kwargs(kwargs_str):
    if len(kwargs_str) == 0:
        return {}
    kwargs = json.loads(kwargs_str)
    assert isinstance(kwargs, dict)
    return kwargs


def create_input(prompt, question):
    messages = []
    for example in prompt:
        messages.extend(dict2problem(example))
    messages.extend(dict2problem(question, False))
    messages = '\n'.join(messages)
    return messages


def dict2problem(dict_input, demo=True):
    problem = "問題: " + dict_input['problem_text']
    choices = dict_input['choices']
    answer = dict_input['answer']
    if len(choices) > 0:
        for choice, label in zip(choices, string.ascii_lowercase):
            problem = problem + '\n' + label + ': ' + choice
        problem = problem + "\n必ずa,b,c,d,eの中からちょうど{}個選んでください。".format(len(answer))
        problem = problem + "\n答え:"
    output = [problem]
    if not demo:
        return output
    output.append(",".join(answer))
    return output
