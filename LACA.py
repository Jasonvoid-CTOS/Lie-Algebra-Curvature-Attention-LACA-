import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Dictionary of hyperparameters for the model
Hyber_parameter = {
    "matrix_size": 1024,          # c dimension for \mathfrak{so}(c)
    "input_feature_dim": 300,   # Pre-trained vector dimensionality (e.g., GloVe, FastText)
    "scaling_factor": 16,       # Base scaling factor for attention
    "num_heads": 1,             # Set to 1 for standard single-head LACA sequence blending
    "embed_dim": 64              # D dimension: Dimensionality of the internal value space
}

class LieAlgebraAttention(nn.Module):
    def __init__(self, num_heads=Hyber_parameter["num_heads"], matrix_size=Hyber_parameter["matrix_size"], scaling_factor=Hyber_parameter["scaling_factor"]):
        super(LieAlgebraAttention, self).__init__()
        if matrix_size < 3:
            raise ValueError("Matrix size must be at least 3 to ensure valid non-commutative Lie algebra structure.")
        
        self.num_heads = num_heads
        self.matrix_size = matrix_size
        self.scaling_factor = scaling_factor
        self.embed_dim = Hyber_parameter["embed_dim"]
        self.input_feature_dim = Hyber_parameter.get("input_feature_dim", 300)
        
        # 1. Standard learnable linear projection layer for embeddings
        self.text_projection = nn.Linear(self.input_feature_dim, self.embed_dim)
        
        # 2. FIXED: Dedicated Value Projection layer creating the standard Value_matrix [B, S, D]
        self.W_v = nn.Linear(self.input_feature_dim, self.embed_dim)
        
        # 3. MLP Instantiation to map embeddings to flattened matrix shapes
        self.mlp_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.matrix_size * self.matrix_size) 
        )

    def Text_to_Embedding(self, dense_text_features):
        return self.text_projection(dense_text_features)

    def MLP_Mapping_Batch(self, embeddings):
        raw_matrices_flat = self.mlp_layer(embeddings)
        target_shape = list(raw_matrices_flat.shape[:-1]) + [self.matrix_size, self.matrix_size]
        raw_matrices = raw_matrices_flat.view(*target_shape)
        return raw_matrices

    def map_skew_symmetric_matrix_function(self, raw_matrices):
        if raw_matrices.shape[-1] != self.matrix_size or raw_matrices.shape[-2] != self.matrix_size:
            raise ValueError(f"Input must end in square matrices of size {self.matrix_size}x{self.matrix_size}.")
        
        skew_symmetric_matrices = 0.5 * (raw_matrices - raw_matrices.transpose(-1, -2))
        
        # Batch verification check
        is_square = skew_symmetric_matrices.shape[-1] == skew_symmetric_matrices.shape[-2]
        is_valid_c = self.matrix_size >= 3
        is_skew = torch.allclose(skew_symmetric_matrices.transpose(-1, -2), -skew_symmetric_matrices, atol=1e-4)
        
        if not (is_square and is_valid_c and is_skew):
            raise ValueError("The resulting matrix tensor contains elements that are not valid skew-symmetric matrices.")
            
        return skew_symmetric_matrices

    def Reformation_of_the_matrix_token_input(self, skew_symmetric_matrices):
        seq_len = skew_symmetric_matrices.shape[-3] 
        
        # Expanding token maps to generate coordinate combinations across a clean matrix grid
        skew_symmetric_matrix_token_i = skew_symmetric_matrices.unsqueeze(-3).expand(*skew_symmetric_matrices.shape[:-3], seq_len, -1, -1, -1)
        skew_symmetric_matrix_token_j = skew_symmetric_matrices.unsqueeze(-4).expand(*skew_symmetric_matrices.shape[:-3], -1, seq_len, -1, -1)
        
        return skew_symmetric_matrix_token_i, skew_symmetric_matrix_token_j, seq_len

    def LACA(self, skew_symmetric_matrix_token_i, skew_symmetric_matrix_token_j, seq_len):
        # Correct Lie Bracket Formula: [A, B] = AB - BA
        Lie_Algebra_difference = (skew_symmetric_matrix_token_i @ skew_symmetric_matrix_token_j) - (skew_symmetric_matrix_token_j @ skew_symmetric_matrix_token_i)
        
        # Stabilized sequence length denominator scaling
        scaled_factor = np.sqrt(self.scaling_factor) * seq_len
        M_scaled = Lie_Algebra_difference / max(scaled_factor, 1.0)
        
        # Map algebra to group structure natively using PyTorch Matrix Exponential
        M_group = torch.linalg.matrix_exp(M_scaled)
        
        # Extract matrix trace along spatial dimensions (-2, -1) resulting in shape [B, S, S]
        Score_LACA = torch.diagonal(M_group, dim1=-2, dim2=-1).sum(-1)
        return Score_LACA

    def forward(self, dense_text_features):
        """
        Full end-to-end forward pass pipeline block.
        """
        # 0. Generate the verified traditional Value_matrix [B, S, D]
        Value_matrix = self.W_v(dense_text_features)
        
        # 1. Project dense vectors to embedding spaces [B, S, D]
        embeddings = self.Text_to_Embedding(dense_text_features)
        
        # 2. Batch-map embeddings directly to raw square matrices [B, S, c, c]
        raw_matrices = self.MLP_Mapping_Batch(embeddings)
        
        # 3. Enforce skew-symmetric structural properties [B, S, c, c]
        skew_matrices = self.map_skew_symmetric_matrix_function(raw_matrices)
        
        # 4. Construct attention coordinate pairs across sequence lengths
        matrix_i, matrix_j, seq_len = self.Reformation_of_the_matrix_token_input(skew_matrices)
        
        # 5. Compute Lie Algebra Curvature Attention maps [B, S, S]
        attention_scores = self.LACA(matrix_i, matrix_j, seq_len)
        
        # 6. (Apply softmax across source sequence length)(if needed softmax)(but for presvering the nature of the Lie algebra,we ignore softmax)
        #attention_weights = torch.softmax(attention_scores, dim=-1)

        confimation = input("Do you want to use the raw attention scores without softmax? (y/n): ").strip().lower()
        if confimation == "y":
            attention_weights = attention_scores
        if confimation == "n":
            attention_weights = torch.softmax(attention_scores, dim=-1)
        # 7. FIXED: Use torch.matmul instead of element-wise multiplication (*)
        # Performs standard sequence blending: [B, S, S] x [B, S, D] -> [B, S, D]
        Output = torch.matmul(attention_weights, Value_matrix)
        return Output

if __name__ == "__main__":
    import re
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # [Your existing setup code: 1. Input Prompt through 5. Model Initialization]
    real_text_sentences = [
        "Lie algebras and matrix attention modeling",
        "Geometric deep learning bends token representations"
    ]
    tokenized_sentences = [re.findall(r'\w+', sentence.lower()) for sentence in real_text_sentences]
    all_words = sorted(list(set([word for sublist in tokenized_sentences for word in sublist])))
    PAD_TOKEN = "<pad>"
    vocab = [PAD_TOKEN] + all_words
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    max_seq_len = max(len(s) for s in tokenized_sentences)
    padded_token_ids = []
    for tokens in tokenized_sentences:
        indices = [word_to_idx[t] for t in tokens]
        indices += [word_to_idx[PAD_TOKEN]] * (max_seq_len - len(indices))
        padded_token_ids.append(indices)
    input_ids = torch.tensor(padded_token_ids, dtype=torch.long)
    embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=Hyber_parameter["input_feature_dim"])
    real_input_features = embedding_layer(input_ids).detach() 
    model = LieAlgebraAttention(num_heads=Hyber_parameter["num_heads"])

    # ======================================================================
    # NEW VISUALIZATION BLOCK: HOOK EXTRACTION & HEATMAP PLOTTING (You may ignore this or keep it for visualize purpose)
    # ======================================================================
    
    # Container to intercept and hold attention weights during the forward pass
    captured_activations = {}

    def attention_hook_fn(module, input_args, output_tensor):
        """Hook to capture the output of step 6.(attention_scores)"""
        captured_activations['attention_weights'] = output_tensor.detach().numpy()

    # Register the hook onto the LACA function's final activation stage inside forward()
    # We can hook onto the forward pass and
    # record the weights matrix right before output blending.
    # To do this cleanly, we record it right inside an adjusted execution block or hook the final module step.
    
    # Alternative: Let's run a modified pass to explicitly extract the weights
    with torch.no_grad():
        # Step-by-step extraction matching your forward pass pipeline
        Value_matrix = model.W_v(real_input_features)
        embeddings = model.Text_to_Embedding(real_input_features)
        raw_matrices = model.MLP_Mapping_Batch(embeddings)
        skew_matrices = model.map_skew_symmetric_matrix_function(raw_matrices)
        matrix_i, matrix_j, seq_len = model.Reformation_of_the_matrix_token_input(skew_matrices)
        attention_scores = model.LACA(matrix_i, matrix_j, seq_len)
        
        attention_weights =attention_scores #without softmax for Lie algebra preservation

    # Reconstruct token lists representing our padded matrix boundaries
    sentence_tokens = []
    for idx, tokens in enumerate(tokenized_sentences):
        padded_tokens = tokens + [PAD_TOKEN] * (max_seq_len - len(tokens))
        sentence_tokens.append(padded_tokens)

    # Set up a matplotlib figure with a subplot row for each sentence
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
    fig.suptitle("Lie Algebra Curvature Attention (LACA) Heatmaps", fontsize=14, weight='bold')

    for batch_idx in range(attention_weights.shape[0]):
        tokens = sentence_tokens[batch_idx]
        matrix_data = attention_weights[batch_idx]
        ax = axes[batch_idx]
        
        # Plot utilizing Seaborn's heatmap layer
        sns.heatmap(
            matrix_data, 
            annot=True, 
            fmt=".3f", 
            cmap="viridis", 
            xticklabels=tokens, 
            yticklabels=tokens, 
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title(f"Sentence {batch_idx + 1}: '{real_text_sentences[batch_idx]}'")
        ax.set_xlabel("Key Tokens (Source Context)")
        ax.set_ylabel("Query Tokens (Target Focus)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    plt.show()



