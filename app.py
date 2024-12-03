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

    start_str = st.text_input("Enter starting letters:", "A")
    num_names = st.slider("How many names to generate?", 1, 20, 5)
    temperature = st.slider("Creativity slider", 0.5, 2.0, 1.0, 0.1)

    if st.button("Generate Names"):
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
            for g, names in genders.items():
                st.write(f"**{g.capitalize()} Names:**")
                for name in names:
                    st.write(f"- {name}")


if __name__ == "__main__":
    app()
