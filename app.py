import streamlit as st
import torch
import pandas as pd
from torch.utils.data import Dataset
from arch import MinimalTransformer  
from utils import sample  

class NameDataset(Dataset):
    def __init__(self, csv_file):
        self.names = pd.read_csv(csv_file, encoding='utf-8')['name'].values
        self.chars = sorted(list(set(''.join(self.names) + ' ')))
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.vocab_size = len(self.chars)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx] + ' '
        encoded_name = [self.char_to_int[char] for char in name]
        return torch.tensor(encoded_name)


m_dataset = NameDataset('m_names.csv')
w_dataset = NameDataset('w_names.csv')

model = torch.load('namesformer_model.pt', map_location=torch.device('cpu'))
model.eval()

def app():
    st.title("Namesformer")

    start_str = st.text_input("Enter starting letters:", "a")
    num_names = st.slider("How many names to generate?", 1, 20, 5)

    valid_chars = set(m_dataset.chars)
    start_str = ''.join([c for c in start_str if c in valid_chars])

    if not start_str:
        st.error("Please enter valid starting letters.")
        return

    if st.button("Generate Names"):
        try:
            results = sample(
                model=model,
                m_dataset=m_dataset,
                w_dataset=w_dataset,
                start_str=start_str,
                max_length=20,
                num_names=num_names,
            )

            st.write(f"**Generated names:**")
            for temp, genders in results.items():
                st.write(f"### {temp}")

                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Man Names")
                    for name in genders['man']:
                        st.write(f"- {name}")
                
                with col2:
                    st.subheader("Woman Names")
                    for name in genders['woman']:
                        st.write(f"- {name}")
        except ValueError as e:
            st.error(str(e))


if __name__ == "__main__":
    app()
