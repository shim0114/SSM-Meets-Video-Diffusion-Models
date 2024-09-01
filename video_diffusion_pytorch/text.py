# Source: video-diffusion-pytorch/video_diffusion_pytorch/video_diffusion_pytorch.py
# This code was copied from lucidrains's video-diffusion-pytorch project, specifically the text.py file.
# For more details and license information, please refer to the original repository:
# Repository URL: https://github.com/lucidrains/video-diffusion-pytorch

import torch
from einops import rearrange

def exists(val):
    return val is not None

# singleton globals

MODEL = None
TOKENIZER = None
BERT_MODEL_DIM = 768

def get_tokenizer(load_pretrained = False):
    if load_pretrained:
        global TOKENIZER
        if not exists(TOKENIZER):
            TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    else:
        try: ### TODO: refactor this ###
            TOKENIZER = torch.load('data/bert-base-cased-tokenizer.pt')
        except:
            TOKENIZER = torch.load('bert-base-cased-tokenizer.pt')
    return TOKENIZER

def get_bert(load_pretrained = False):
    if load_pretrained:
        if not exists(MODEL):
            MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
            if torch.cuda.is_available():
                MODEL = MODEL.cuda()
    else:
        try: ### TODO: refactor this ###
            MODEL = torch.load('data/bert-base-cased.pt')
        except:
            MODEL = torch.load('bert-base-cased.pt')
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()
    return MODEL

# tokenize

def tokenize(texts, add_special_tokens = True):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    tokenizer = get_tokenizer()

    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens = add_special_tokens,
        padding = True,
        return_tensors = 'pt'
    )

    token_ids = encoding.input_ids
    return token_ids

# embedding function

@torch.no_grad()
def bert_embed(
    token_ids,
    return_cls_repr = False,
    eps = 1e-8,
    pad_id = 0.
):
    model = get_bert()
    mask = token_ids != pad_id

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    outputs = model(
        input_ids = token_ids,
        attention_mask = mask,
        output_hidden_states = True
    )

    hidden_state = outputs.hidden_states[-1]

    if return_cls_repr:
        return hidden_state[:, 0]               # return [cls] as representation

    if not exists(mask):
        return hidden_state.mean(dim = 1)

    mask = mask[:, 1:]                          # mean all tokens excluding [cls], accounting for length
    mask = rearrange(mask, 'b n -> b n 1')

    numer = (hidden_state[:, 1:] * mask).sum(dim = 1)
    denom = mask.sum(dim = 1)
    masked_mean =  numer / (denom + eps)
    return masked_mean

if __name__ == "__main__":
    tokenizer = get_tokenizer(load_pretrained=True)
    model = get_bert(load_pretrained=True)
    torch.save(model, 'bert-base-cased.pt')
    torch.save(tokenizer, 'bert-base-cased-tokenizer.pt')
    