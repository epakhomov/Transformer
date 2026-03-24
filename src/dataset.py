import random
import torch
from torch.utils.data import Dataset
import argparse

"""

Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
optimizing the model to predict the question, "Where was...".

Note that the NameDataset should take the pretraining_dataset defined in run.py
as an input. This is to allow the vocab specification of the NameDataset to be
the same as that of the pretraining dataset.

You don't need to implement anything in NameDataset.
"""

class NameDataset(Dataset):
    def __init__(self, data, pretraining_dataset):
        self.MASK_CHAR = pretraining_dataset.MASK_CHAR # the doublequestionmark character, for mask
        self.PAD_CHAR = pretraining_dataset.PAD_CHAR # the empty square character, for pad
        self.itos = pretraining_dataset.itos 
        self.stoi = pretraining_dataset.stoi 
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) - 1

    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR*(self.block_size - len(x))
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]
        
        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y

"""


Here is a custom class that yields examples of a simplified span corruption objective.


Masking Specification

The __getitem__ function takes an index and returns a data point (x, y) where
x and y are Long tensors of length self.block_size. x encodes the input
sequence, and y encodes the output sequence.


"""
class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.max_context_size = int(block_size*3/4)
        self.masking_percent = 0.25
        self.vocab_size = vocab_size
        self.data = data.split('\n')
        self.item = None

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):

        ### TODO:
        ### [part e]: see spec above

        ### START CODE HERE
        # 0. Get the document at the given index
        document = self.data[idx]

        # 1. Randomly truncate the document
        # Choose a random length between 4 and block_size*3/4
        min_len = 4
        max_len = min(len(document), int(self.block_size * 3 / 4))
        trunc_len = random.randint(min_len, max_len)
        truncated_doc = document[:trunc_len]

        # 2. Split into prefix, masked_content, and suffix
        # First determine the length of masked_content
        # Target average is 1/4 of truncated length, so we'll use random.randint
        # around that value to get variable lengths
        target_mask_len = max(1, int(len(truncated_doc) * 0.25))
        mask_len = random.randint(1, target_mask_len * 2)  # Allow variation around target
        mask_len = min(mask_len, len(truncated_doc) - 1)  # Ensure we leave at least 1 char for prefix

        # Choose a random start point for the mask
        mask_start = random.randint(1, len(truncated_doc) - mask_len)

        # Split the document
        prefix = truncated_doc[:mask_start]
        masked_content = truncated_doc[mask_start:mask_start + mask_len]
        suffix = truncated_doc[mask_start + mask_len:]

        # 3. Construct the masked string with format:
        # [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] MASK_CHAR [pads]
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content + self.MASK_CHAR

        # Add padding to reach block_size
        num_pads = self.block_size - len(masked_string)
        masked_string = masked_string + self.PAD_CHAR * num_pads

        # 4. Create input and output strings
        # Input is masked_string[:-1], output is masked_string[1:]
        inp = masked_string[:-1]
        out = masked_string[1:]

        # 5. Encode to tensors using the vocabulary
        x = torch.tensor([self.stoi[c] for c in inp], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in out], dtype=torch.long)

        return x, y
        ### END CODE HERE

        raise NotImplementedError

"""
Code under here is strictly for  debugging purposes
"""
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: namedata, charcorruption.",
            choices=["namedata", "charcorruption"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(open('./../data/wiki.txt', encoding='utf-8').read(), 128) 
        # Make the name dataset
        name_dataset = NameDataset(open('./../data/birth_places_train.tsv', encoding='utf-8').read(),
                corruption_dataset)
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
        pass
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(open('./../data/wiki.txt', encoding='utf-8').read(), 128) 
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                .format(args.dataset_type))

