import torch
import torch.nn as nn

def train(model, dataloader, gender, epochs=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for input_seq, target_seq in dataloader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            optimizer.zero_grad()
            output = model(input_seq, gender=gender)
            loss = criterion(output.transpose(1, 2), target_seq)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} ({gender}), Loss: {total_loss / len(dataloader)}")


def sample(model, m_dataset, w_dataset, start_str='A', max_length=20, num_names=5):
    temperatures = [0.5, 1.5]
    device = next(model.parameters()).device

    print(f"{'Confidence':<20} {'Man Names':<30} {'Woman Names':<30}")
    print("-" * 85)

    for temp in temperatures:
        m_names = []
        w_names = []

        for _ in range(num_names):
            model.eval()
            with torch.no_grad():
                # Generate m names
                chars = [m_dataset.char_to_int[c] for c in start_str]
                input_seq = torch.tensor(chars, device=device).unsqueeze(0)
                output_name = start_str
                for _ in range(max_length - len(start_str)):
                    output = model(input_seq, gender="m")
                    logits = output[0, -1] / temp
                    probabilities = torch.softmax(logits, dim=0)
                    next_char_idx = torch.multinomial(probabilities, 1).item()
                    next_char = m_dataset.int_to_char[next_char_idx]
                    if next_char == ' ':
                        break
                    output_name += next_char
                    input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]], device=device)], dim=1)
                m_names.append(output_name)

        for _ in range(num_names):
            model.eval()
            with torch.no_grad():
                chars = [w_dataset.char_to_int[c] for c in start_str]
                input_seq = torch.tensor(chars, device=device).unsqueeze(0)
                output_name = start_str
                for _ in range(max_length - len(start_str)):
                    output = model(input_seq, gender="w")
                    logits = output[0, -1] / temp
                    probabilities = torch.softmax(logits, dim=0)
                    next_char_idx = torch.multinomial(probabilities, 1).item()
                    next_char = w_dataset.int_to_char[next_char_idx]
                    if next_char == ' ':
                        break
                    output_name += next_char
                    input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]], device=device)], dim=1)
                w_names.append(output_name)

        confidence_label = "Higher Confidence" if temp == 0.5 else "More Creative"
        print(f"{confidence_label:<20}")
        for m_name, w_name in zip(m_names, w_names):
            print(f"{'':<20} {m_name:<30} {w_name:<30}")
        print()

