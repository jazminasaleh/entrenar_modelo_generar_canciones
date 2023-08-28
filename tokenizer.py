import string

#abrir el archivo en modo lectura
f = open("canciones.txt", "r", encoding='utf-8')
text = f.read()
all_characters = string.printable + "ñÑáÁéÉíÍóÓúÚ¿?()¡"

#Tokeniza y convertir los textos en secuencias numericas y viceversa
#text_to_seq: cadena de texto a secuencia numerica
#seq_to_text: secuencia numerica a textos
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

#prueba de la tokenizacion
tokenizer = Tokenizer()
print(tokenizer.n_characters)
print(tokenizer.text_to_seq('señor, ¿que tal?'))
text_encoded = tokenizer.text_to_seq(text)