
import matplotlib.pyplot as plt
import torch
import numpy as np

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.image_counter = 0

    def sanity_check(self, sentence, block_size, layer=None, head=None):
        # Encode the sentence using the tokenizer
        print(f'\nEncoding Sentence: {sentence}')
        wordids = self.tokenizer.encode(sentence) # (1,len(tokens))
        # print(f'Token Indices: {wordids}')

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))  # (1, block_size)
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0) # (1, 1, block_size)

        # Generate tokens with padding labels
        list_of_tokens = [self.tokenizer.itos.get(index) for index in padded_sentence]
        # print(f'Token list: {list_of_tokens}')

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        _,  attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        # Display the number of attention maps
        print("Number of attention maps = Layers * heads = ", len(attn_maps)*attn_maps[0].squeeze(0).shape[0])

        map_counter=0
        # Visualize and save the attention maps
        for i, amap in enumerate(attn_maps): #for each layer
            for j in range(amap.size(1)):
                if (i+1 == layer and j+1 == head) or layer == None or head == None:
                    map_counter += 1
                    attn_map = amap[0,j].detach().cpu().numpy()
                    total_prob_over_rows = torch.sum(torch.tensor(attn_map), dim=1)
                    if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                        print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                        print("Total probability over rows:", total_prob_over_rows.detach().numpy())
        
                    # Create a heatmap of the attention map
                    fig, ax = plt.subplots(figsize=(8, 8))  # Increase figure size if needed
                    cax = ax.imshow(attn_map, cmap='hot', interpolation='nearest')
                    ax.xaxis.tick_top()
                    fig.colorbar(cax, ax=ax)
                    plt.title(f"Attention Map for Layer #{i+1}, Head #{j+1}")
                    
                    # Set word labels on x and y axes
                    ax.set_xticks(range(len(list_of_tokens)))
                    ax.set_xticklabels(list_of_tokens, rotation=90, fontsize=8)  # Rotate for readability and adjust fontsize
                    ax.set_yticks(range(len(list_of_tokens)))
                    ax.set_yticklabels(list_of_tokens, fontsize=8)  # Adjust fontsize
                    
                    plt.tight_layout()  # Adjust layout for better label visibility
                    
                    # Save and show the plot
                    self.image_counter += 1
                    plt.savefig(f"attention_map_{i+1}{j+1}-{self.image_counter}.png")
                    plt.show()

                
            # att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array

            # Check if the attention probabilities sum to 1 over rows
            
