import streamlit as st
import torch
import pandas as pd

st.write("""# Summerize your text""")


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("tokenizer")

model = AutoModelForSeq2SeqLM.from_pretrained("pegasus_summery_model")

text_input = st.text_area("text to summerize")

if text_input:
    
    tokenized_text = tokenizer.encode_plus(
        str(text_input),
        return_attention_mask= True,
        return_tensors='pt'
    )
    
    generated_token = model.generate(
        input_ids = tokenized_text['input_ids'],
        attention_mask = tokenized_text["attention_mask"],
        use_cache=True,)
    
    
    pred = [tokenizer.decode(token_ids=ids, skip_special_tokens=True)for ids in generated_token]
    
    st.write("## Summerized Text")
    st.write(" ".join(pred))