import string
import torch
from tqdm import tqdm
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)
all_characters = string.printable + "ñÑáÁéÉíÍóÓúÚ¿?()¡"
class Tokenizer():
    def __init__(self):
        self.all_characters = all_characters
        self.n_characters = len(self.all_characters)

    def text_to_seq(self, string):
        seq = []
        for c in range(len(string)):
            try:
                seq.append(self.all_characters.index(string[c]))
            except:
                continue
        return seq

    def seq_to_text(self, seq):
        text = ''
        for c in range(len(seq)):
            text += self.all_characters[seq[c]]
        return text

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    seed_text = data['seed_text']

    #abrir el archivo en modo lectura
    f = open("canciones.txt", "r", encoding='utf-8')
    text = f.read()

    

    #tokenizacion
    tokenizer = Tokenizer()
    text_encoded = tokenizer.text_to_seq(text)

    #Se dividen los datos en un conjunto de entrenamiento y un conjunto de prueba
    #train: se crea el conjunto de entenameinto toamno los primeros elmentos de text_encoded
    #test: se crea un conjuntode prueba, toamndo los elemnetos restantes de text_encoded
    train_size = int(len(text_encoded) * 0.8) 
    train = text_encoded[:train_size]
    test = text_encoded[train_size:]
    print('este es test')
    print(len(test))

    #divide una secuencia en ventanass y estas se almacenan en text_windows
    #Para entrenar el modleo se neceistas secuencias de textos de una longitud detemrinada
    def windows(text, window_size = 100):
        start_index = 0
        end_index = len(text) - window_size
        text_windows = []
        while start_index < end_index:
          text_windows.append(text[start_index:start_index+window_size+1])
          start_index += 1
        return text_windows

    text_encoded_windows = windows(text_encoded)
    print(tokenizer.seq_to_text((text_encoded_windows[0])))
    print()
    print(tokenizer.seq_to_text((text_encoded_windows[1])))
    print()
    print(tokenizer.seq_to_text((text_encoded_windows[2])))

    #se encerga de dar todos los caracteres excepto el ultimo, ya que este la red debera predecirlo
    class CharRNNDataset(torch.utils.data.Dataset):
      def __init__(self, text_encoded_windows, train=True):
        self.text = text_encoded_windows
        self.train = train

      def __len__(self):
        return len(self.text)

      def __getitem__(self, ix):
        if self.train:
          return torch.tensor(self.text[ix][:-1]), torch.tensor(self.text[ix][-1])
        return torch.tensor(self.text[ix])
      
    train_text_encoded_windows = windows(train)
    test_text_encoded_windows = windows(test)

    dataset = {
        'train': CharRNNDataset(train_text_encoded_windows),
        'val': CharRNNDataset(test_text_encoded_windows)
    }

    dataloader = {
        'train': torch.utils.data.DataLoader(dataset['train'], batch_size=512, shuffle=True, pin_memory=True),
        'val': torch.utils.data.DataLoader(dataset['val'], batch_size=2048, shuffle=False, pin_memory=True),
    }

    print(len(dataset['train']))
    print(len(dataset['val']))

    input, output = dataset['train'][0]
    print(tokenizer.seq_to_text(input))

    print(tokenizer.seq_to_text([output]))

    #input_size: cantidad de caracteres unicos
    #embedding_size: dimension del espacio incrustacion
    #hidden_size: numero de unidades ocultas en la capa oculta de la RNN
    #num_layers: nunmero de capas de la RNN

    class CharRNN(torch.nn.Module):
      def __init__(self, input_size, embedding_size=128, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, input_size)

      def forward(self, x):
        x = self.encoder(x)
        x, h = self.rnn(x)         
        y = self.fc(x[:,-1,:])
        return y
      
    model = CharRNN(input_size=tokenizer.n_characters)
    outputs = model(torch.randint(0, tokenizer.n_characters, (64, 50)))
    outputs.shape
    torch.Size([64, 114])

    #entrenamiento

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(model, dataloader, epochs=10):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1, epochs+1):
            model.train()
            train_loss = []
            bar = tqdm(dataloader['train'])
            for batch in bar:
                X, y = batch
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_hat = model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                bar.set_description(f"loss {np.mean(train_loss):.5f}")
            bar = tqdm(dataloader['val'])
            val_loss = []
            with torch.no_grad():
                for batch in bar:
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    y_hat = model(X)
                    loss = criterion(y_hat, y)
                    val_loss.append(loss.item())
                    bar.set_description(f"val_loss {np.mean(val_loss):.5f}")
            print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} val_loss {np.mean(val_loss):.5f}")
            if epoch == epochs:
              torch.save(model.state_dict(), "modelo_entrenado_ultima_epoca120_ventanas200.pth")

    def predict(model, X):
      model.eval() 
      with torch.no_grad():
        X = torch.tensor(X).to(device)
        pred = model(X.unsqueeze(0))
        return pred
      
    model = CharRNN(input_size=tokenizer.n_characters)

    #mirar si el modelo ya esta entrenado
    if os.path.exists("modelo_entrenado_ultima_epoca120_ventanas200.pth"):
        print('El modelo ya está entrenado.')
        try:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load("modelo_entrenado_ultima_epoca120_ventanas200.pth"))
            else:
                model.load_state_dict(torch.load("modelo_entrenado_ultima_epoca120_ventanas200.pth", map_location=torch.device('cpu')))
            model.eval()
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
    else:
        print('Hay que entrenar el modelo')
        fit(model, dataloader, epochs=150)

    #coifica el contexto inicial
    X_new = seed_text
    X_new_encoded = tokenizer.text_to_seq(X_new)
    y_pred = predict(model, X_new_encoded)
    y_pred = torch.argmax(y_pred, axis=1)[0].item()
    tokenizer.seq_to_text([y_pred])

    #generar las 300 caracteres
    temp=1
    for i in range(300):
      X_new_encoded = tokenizer.text_to_seq(X_new[-100:])
      y_pred = predict(model, X_new_encoded)
      y_pred = y_pred.view(-1).div(temp).exp()
      top_i = torch.multinomial(y_pred, 1)[0]
      predicted_char = tokenizer.all_characters[top_i]
      X_new += predicted_char

    print('este es el texto generado:')
    print(X_new)
    print('Este es el corregido')
    #chatgpt
    openai.api_key = "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"
    prompt = "me podrias corregir este texto: "+ X_new
    mensaje = [
    {'role': 'user', 'content': prompt}
    ]   
    textoCorregido = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= mensaje,
        temperature = 0.8,
        max_tokens=300
    )

    print(textoCorregido.choices[0].message["content"])


    #Guradr el texto en el archivo
    with open("texto_generado.txt", "w", encoding="utf-8") as output_file:
        output_file.write(X_new)
        print("Texto generado guardado en 'texto_generado.txt'")

    return jsonify({'generated_text': X_new, 'corrected_text': textoCorregido.choices[0].message["content"]})

if __name__ == '__main__':
    app.run()