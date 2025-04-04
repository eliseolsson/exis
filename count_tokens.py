from datasets import load_from_disk
data = load_from_disk("corpus_ds")

from transformers import AutoModel, AutoTokenizer
MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

tokens =  tokenizer(data["combined_text"], padding=False, truncation=False, max_length=5192)

import pickle
tokens_inputs = tokens["input_ids"] 

ntokens = [len(tokens) for tokens in tokens_inputs]
with open("number_tokens.pkl", "wb") as f:
    pickle.dump(ntokens, f)

import matplotlib.pyplot as plt
plt.hist(ntokens)
plt.show() 

print(f"Mean: {sum(ntokens)/len(ntokens)}")