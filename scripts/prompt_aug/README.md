# Prompt Aug.

you can augment `action_captions.json` through an LLM of your choice from hugging face:
1. it's *strongly recomended to create a separate conda environment solely for the dependancies of the following script*, as it has conflicts with MDM dependancies. Once obtained the augmented
2. Add your [identification token](https://huggingface.co/settings/tokens) from hugging face running `huggingface-cli login` on your terminal and pasting your access token.
3. Make sure you have permission to access the requested model, otherwise, make a requeste on [huggingface.co](https://huggingface.co/)

```
python3 action_2_caption.py \
  --dataset NTU60 \
  --model mistral-7b \
  --n 5 \
  --training
```

consider that the used .json within MDM model is always the one named `<dataset>/class\_captions.json`