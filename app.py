import streamlit as st
import torch
import pandas as pd
from torch.utils.data import Dataset
from arch import MinimalTransformer  
from utils import train, sample  

class NameDataset(Dataset):
    def __init__(self, csv_file):
        self.names = pd.read_csv(csv_file)['name'].values
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


m_dataset = NameDataset('man_names.csv')
w_dataset = NameDataset('woman_names.csv')

model = MinimalTransformer(
    vocab_size=man_dataset.vocab_size,
    embed_size=128,
    num_heads=8,
    forward_expansion=4,
)
model = torch.load('namesformer_model.pt', map_location=torch.device('cpu'))
model.eval()

def app():
    st.title("Namesformer")

    start_str = st.text_input("Enter starting letters:", " ")
    gender = st.selectbox("Choose gender:", ["man", "woman"])
    num_names = st.slider("How many names to generate?", 1, 20, 5)
    temperature = st.slider("Creativity slider ", 0.5, 2.0, 1.0, 0.1)

    if st.button("Generate Names"):
        dataset = man_dataset if gender == "man" else woman_dataset
        st.write(f"**Generated {gender} names:**")
        for _ in range(num_names):
            name = sample(
                model=model,
                man_dataset=man_dataset,
                woman_dataset=woman_dataset,
                start_str=start_str,
                max_length=20,
                num_names=1,
            )
            st.write(f"- {name}")


if __name__ == "__main__":
    app()
