import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, AdamW
from torchinfo import summary
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerModelwithCLS, TransformerDecoder, TransformerModelwithCLS_AliBi
from utilities import Utilities

import argparse  # Argument parsing

import matplotlib.pyplot as plt

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match the numbers mentioned in the assignment description """

# Hyperparameters
embed_dim = 64        # Embedding dimension
num_heads = 2         # Number of attention heads
num_layers = 4        # Number of transformer layers
num_classes = 3       # Number of classes (Barack Obama, George W. Bush, George H. Bush)
n_input = 64          # Input size for the classifier, should match the embedding size of the transformer
ff_dim = 100          # Feedforward dimension (hidden dim)
max_len = 32          # Maximum sequence length or block_size
batch_size = 16       # Batch size
dropout = 0.1         # Dropout rate
num_epochs = 15       # Number of training epochs
learning_rate = 1e-3  # Learning rate
eval_interval = 100   # How often to evaluate train and test perplexity during training
eval_iters = 200      # Number of iterations to evaluate perplexity on the test set
max_iters = 500       # For language modeling, we can process all the batches for the entire dataset, but that takes a while, 
                      # so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this 
                      # is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very 
                      # small dataset.


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
            
        filepath = os.path.join(directory, filename)
        # Check if the path is a file before trying to open it
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :max_len]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, max_len - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def trainClassifier(train_loader, model, loss_fn, optimizer, device=None):
    model.train()  # Set the model to training mode
    total_loss, total_correct = 0.0, 0.0

    if device is None:
        device = globals().get('device')  # Retrieve global device if not provided

    for batch in train_loader:
        inputs, labels = batch  # Get input data and labels from the batch

        # Move inputs and labels to the appropriate device (e.g., GPU if available)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass through the model
        logits, _ = model(inputs)  # Get logits and attention maps
        
        # Compute the loss
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Compute correct predictions
        _, predicted = torch.max(logits.data, 1)
        total_correct += (predicted == labels).sum().item()

        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

    avg_loss = total_loss / len(train_loader)  # Average loss for the epoch
    accuracy = (100 * total_correct / len(train_loader.dataset))  # Accuracy for the epoch
    return avg_loss, accuracy  

def evaluateClassifier(test_loader, model, loss_fn, device=None):
    model.eval()  # Set the model to evaluation mode
    total_loss, total_correct = 0.0, 0.0

    if device is None:
        device = globals().get('device')  # Retrieve global device if not provided

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch  # Get input data and labels from the batch

            # Move inputs and labels to the appropriate device (e.g., GPU if available)
            inputs, labels = inputs.to(device), labels.to(device)

            # Get logits, ignore attention maps
            logits, _ = model(inputs) 

            # Compute Loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()  # Convert loss to scalar

            # Compute correct predictions
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)  # Average loss for the epoch
    accuracy = (100 * total_correct / len(test_loader.dataset))  # Accuracy for the epoch
    return avg_loss, accuracy  

def compute_perplexity(model, loss_fn, data_loader, eval_iters=200, device=None):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    if device is None:
        device = globals().get('device')  # Retrieve global device if not provided
        
    model.eval()
    total_loss = 0.0
    num_losses = 0

    with torch.no_grad():  # Disable gradient calculation
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            logits, _ = model(X)  # Get logits from the model, ignore attnmaps
            loss = loss_fn(logits.view(-1, logits.size(-1)), Y.view(-1))  # Compute loss
            total_loss += loss.item()
            num_losses += 1
            if num_losses >= eval_iters:
                break

    # Calculate mean loss and perplexity
    mean_loss = total_loss / num_losses if num_losses > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(mean_loss)).item()  # Calculate perplexity as exp(mean loss)

    model.train()
    return perplexity

    
def trainAndEvaluateLM(train_loader, test_loader, model, loss_fn, optimizer, device=None):
    model.train()  # Set the model to training mode
    total_loss = []

    if device is None:
        device = globals().get('device')  # Retrieve global device if not provided

    for iteration, (x, y) in enumerate(train_loader):
        if iteration >= max_iters:
            break
            
        # Move inputs and labels to the appropriate device (e.g., GPU if available)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass through the model
        logits, _ = model(x)  # Get logits and attention maps

        # Reshape logits and labels for the loss function
        logits = logits.view(-1, logits.size(-1)) # (batch_size, seq_len, vocab_size) => (batch_size*seq_len, vocab_size)
        y = y.view(-1)                       # (batch_size, seq_len) => (batch_size*seq_len,)
        
        # Backward pass and optimization
        loss = loss_fn(logits, y)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        if (iteration+1) % 100 == 0:  # Evaluate perplexity at every 100 iterations
            # Calculate Train Perplexity
            train_perplexity = compute_perplexity(model, loss_fn, train_loader)
            test_perplexity = compute_perplexity(model, loss_fn, test_loader)
            print(f'Iteration [{iteration+1}/{max_iters}], Training Perplexity: {train_perplexity:.2f}, Test Perplexity: {test_perplexity:.2f}')

def plot_accuracy(*args, **kwargs):
    # Create a Figure
    filename=None
    plt.figure(figsize=(8, 6))

    # Plot lines
    for arg in args:
        plt.plot(arg['accuracy'], label=arg['label'])

    plt.xlabel('Epochs')
    
    for key, value in kwargs.items():
        if key == 'type' and value == 'train':
            plt.ylabel('Training Accuracy')
        elif key == 'type' and value == 'test':
            plt.ylabel('Testing Accuracy')
        elif key == 'title':
            plt.title(value)
        elif key == 'file':
            filename = value
            
    plt.legend()
    plt.grid()

    plt.savefig(filename)
    print(f"\nPlot saved as {filename}")

def main():

    #Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    #Parse the command-line arguments
    args = parser.parse_args()

    # -------------------------------------------------------------------------------------------
    # Common
    
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    global dropout, num_epochs, learning_rate
    
    # -------------------------------------------------------------------------------------------
    # PART 1
    
    if args.model == "part1" or args.model == "all":
        # Load the training data
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)
    
        # Initialize the model, loss function, and optimizer
        model = TransformerModelwithCLS(tokenizer.vocab_size, embed_dim, num_heads, ff_dim, num_layers, num_classes, max_len, dropout=0.1)
        loss_fn = nn.CrossEntropyLoss()  # Loss function for multi-class classification
        optimizer = Adam(model.parameters(), lr=learning_rate)
    
        # Training loop
        for epoch in range(num_epochs):
            train_loss, train_accuracy = trainClassifier(train_CLS_loader, model, loss_fn, optimizer)
            test_loss, test_accuracy = evaluateClassifier(test_CLS_loader, model, loss_fn)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}')
    
        # Sanity check
        utilities = Utilities(tokenizer, model)
        sentence = 	"When one nation pursues a nuclear weapon, the risk of nuclear attack rises for all nations."
        utilities.sanity_check(sentence, max_len, layer=4, head=1)
        
        sentence = 	"The level of world cooperation and condemnation of Iraq is unprecedented."
        utilities.sanity_check(sentence, max_len, layer=3, head=2)  

        # # Calculating parameters
        # dummy_input = torch.randint(0, 5755, (16, 32)).long()
        # try:
        #     output = model(dummy_input)  # Test the model with the dummy input
        #     print("Model output shape:", output.shape)
        # except Exception as e:
        #     print("Error during forward pass:", e)

        # # summarize the model parameters
        # summary(model, input_data=dummy_input)

    # -------------------------------------------------------------------------------------------
    # PART 2

    if args.model == "part2" or args.model == "all":
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  max_len)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        
        # Training and Testing LM for 0: Barack Obama
        print('\n\nTraining and Testing LM for 0: Barack Obama')
        # Initialize the model, loss function, and optimizer
        model = TransformerDecoder(tokenizer.vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, dropout=0.1)
        loss_fn = nn.CrossEntropyLoss() 
        optimizer = Adam(model.parameters(), lr=learning_rate)
        inputfile = "speechesdataset/test_LM_obama.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText,  max_len)
        test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size)
        trainAndEvaluateLM(train_LM_loader, test_LM_loader, model, loss_fn, optimizer)        

        # Training and Testing LM for 1: George W. Bush
        print('\n\nTraining and Testing LM for 1: George W. Bush')
        # Initialize the model, loss function, and optimizer
        model = TransformerDecoder(tokenizer.vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, dropout=0.1)
        loss_fn = nn.CrossEntropyLoss() 
        optimizer = Adam(model.parameters(), lr=learning_rate)
        inputfile = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText,  max_len)
        test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size)
        trainAndEvaluateLM(train_LM_loader, test_LM_loader, model, loss_fn, optimizer)

        # Training and Testing LM for 2: George H. Bush
        print('\n\nTraining and Testing LM for 2: George H. Bush')
        # Initialize the model, loss function, and optimizer
        model = TransformerDecoder(tokenizer.vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, dropout=0.1)
        loss_fn = nn.CrossEntropyLoss() 
        optimizer = Adam(model.parameters(), lr=learning_rate)
        inputfile = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText,  max_len)
        test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size)
        trainAndEvaluateLM(train_LM_loader, test_LM_loader, model, loss_fn, optimizer)

        # Sanity Check and Attention Maps
        utilities = Utilities(tokenizer, model)
        sentence = 	"More of you have lost your homes and even more are watching your home values plummet."
        utilities.sanity_check(sentence, max_len, layer=4, head=1)
        
        sentence = 	"For peace to come, it is time for them, and all of us, to live up to our responsibilities."
        utilities.sanity_check(sentence, max_len, layer=3, head=2)

        # # Calculating parameters
        # dummy_input = torch.randint(0, 5755, (16, 32)).long()
        # try:
        #     output = model(dummy_input)  # Test the model with the dummy input
        #     print("Model output shape:", output.shape)
        # except Exception as e:
        #     print("Error during forward pass:", e)

        # # summarize the model parameters
        # summary(model, input_data=dummy_input)
    
    # -------------------------------------------------------------------------------------------
    # PART 2
    if args.model == "part3" or args.model == "all":

        # We will train these models for 50 epochs
        num_epochs = 100
        learning_rate = 1e-4
        weight_decay=1e-1
        dropout=0.1
        scaling_factor = 1.1 # used to scale the distance bias in alibi encoding

        # print(num_epochs, learning_rate, dropout)
        
        # Load the training data
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

        
        print("Comparing performance of a Transformer Encoder Model with FF Classifier on Absolute Positional Encoding (Sinusoidal) vs AliBi (Attention with Linear Biases)")
        print()
        print("Using Absolute Positional Encoding:")
        # Initialize the model, loss function, and optimizer
        model1 = TransformerModelwithCLS(tokenizer.vocab_size, embed_dim, num_heads, ff_dim, num_layers, num_classes, max_len, dropout)
        loss_fn = nn.CrossEntropyLoss()  # Loss function for multi-class classification
        optimizer = AdamW(model1.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Training loop
        train_accuracies_ape = []
        test_accuracies_ape = []
        for epoch in range(num_epochs):
            train_loss, train_accuracy = trainClassifier(train_CLS_loader, model1, loss_fn, optimizer)
            test_loss, test_accuracy = evaluateClassifier(test_CLS_loader, model1, loss_fn)
            train_accuracies_ape.append(train_accuracy)
            test_accuracies_ape.append(test_accuracy)
            if (epoch+1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Training Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}')
        
        print()
        print("Using AliBi (Attention with Linear Biases):")
        # Initialize the model, loss function, and optimizer
        model2 = TransformerModelwithCLS_AliBi(tokenizer.vocab_size, embed_dim, num_heads, ff_dim, num_layers, num_classes, max_len, dropout, alibi=True, scaling_factor=scaling_factor)
        loss_fn = nn.CrossEntropyLoss()  # Loss function for multi-class classification
        optimizer = AdamW(model2.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_accuracies_alibi = []
        test_accuracies_alibi = []
        # Training loop
        for epoch in range(num_epochs):
            train_loss, train_accuracy = trainClassifier(train_CLS_loader, model2, loss_fn, optimizer)
            test_loss, test_accuracy = evaluateClassifier(test_CLS_loader, model2, loss_fn)
            train_accuracies_alibi.append(train_accuracy)
            test_accuracies_alibi.append(test_accuracy)
            if (epoch+1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Training Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}')        

        
        # Sanity check
        # Absolute positional encoding
        utilities1 = Utilities(tokenizer, model1)
        sentence = 	"Because next week, in Minnesota, the same party that brought you two terms of George Bush and Dick Cheney will ask this country for a third."
        utilities1.sanity_check(sentence, max_len, layer=3, head=2)
        

        # AliBi
        utilities2 = Utilities(tokenizer, model2)
        utilities2.image_counter = utilities1.image_counter
        sentence = 	"Because next week, in Minnesota, the same party that brought you two terms of George Bush and Dick Cheney will ask this country for a third."
        utilities2.sanity_check(sentence, max_len, layer=3, head=2)

        plot_accuracy({'accuracy': train_accuracies_ape, 'label': 'Absolute Positional Encoding'}, 
                      {'accuracy': train_accuracies_alibi, 'label': 'AliBi'},
                     type='train', title='Training Accuracy for Abs Positional Encoding vs AliBi',
                     file='Figure-train.png')
        
        plot_accuracy({'accuracy': test_accuracies_ape, 'label': 'Absolute Positional Encoding'}, 
                      {'accuracy': test_accuracies_alibi, 'label': 'AliBi'},
                     type='test', title='Test Accuracy for Abs Positional Encoding vs AliBi',
                     file='Figure-test.png')
                    
if __name__ == "__main__":
    main()