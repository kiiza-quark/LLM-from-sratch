from tokenizer import Tokenizer

tokenizer = Tokenizer()

# Create the training data
training_data = '. '.join([
'cats rule the world',
'dogs are the best',
'elephants have long trunks',
'monkeys like bananas',
'pandas eat bamboo',
'tigers are dangerous',
'zebras have stripes',
'lions are the kings of the savannah',
'giraffes have long necks',
'hippos are big and scary',
'rhinos have horns',
'penguins live in the arctic',
'polar bears are white'
])

with open("makedata/data.txt", 'r') as file:
    training_data = '. '.join([file.readlines()])
        
# Tokenize the training data
tokenized_training_data = tokenizer.tokenize(training_data)
# Add padding to the left, to make sure all parts of the sequence are being trained

for _ in range(20):
# Prepend padding tokens
    tokenized_training_data.insert(0, tokenizer.character_to_token('<pad>'))