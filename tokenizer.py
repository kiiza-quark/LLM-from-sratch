class Tokenizer:
    def __init__(self):
        self.dictionary = {}
        self.reverse_dictionary = {}
        
        # Add the padding token
        self.__add_to_dict('<pad>')

    # Add characters and numbers to the dictionary
        for i in range(10):
            self.__add_to_dict(str(i))
        for i in range(26):
            self.__add_to_dict(chr(ord('a') + i))

        special_characters = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '=', '[', ']', '{', '}', '|', '\\', ';', ':', '\'',
                                '\"', ',', '.', '<', '>', '/', '?', '`', '~']
        mathematical_symbols = ['+', '-', '*', '/', '=', '<', '>', '≠', '≈', '≡', '≤', '≥', '√', 'π', '∞']
        capital_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                            'V', 'W', 'X', 'Y', 'Z']
        all_symbols = special_characters + mathematical_symbols + capital_letters
        
        for c in all_symbols:
            self.__add_to_dict(c)
        # Add space and punctuation to the dictionary
        # self.__add_to_dict('.')
        # self.__add_to_dict(' ')
        # self.__add_to_dict(':')

    def __add_to_dict(self, character):
        if character not in self.dictionary:
            self.dictionary[character] = len(self.dictionary)
            self.reverse_dictionary[self.dictionary[character]] = character
    
    def tokenize(self, text):
        return [self.dictionary[c] for c in text]
    
    def character_to_token(self, character):
        return self.dictionary[character]
    
    def token_to_character(self, token):
        return self.reverse_dictionary[token]
    
    def size(self):
        return len(self.dictionary)
    


tokenizer = Tokenizer()
print(tokenizer.size()) #103