import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import math
import matplotlib.pyplot as plt
import sqlite3
import json
import mlx.data as dx
import wandb



class Identity(nn.Module):
    """Identity Module."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Squeeze(nn.Module):
    """Squeeze Module."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class SuperLinear(nn.Module):
    """SuperLinear Layer: Implements Neuron-Level Models (NLMs) for the CTM."""
    def __init__(self, in_dims, out_dims, N):
        super().__init__()
        self.in_dims = in_dims
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))

    def forward(self, x):
            out = torch.einsum('BDM,MHD->BDH', x, self.w1) + self.b1
            out = out.squeeze(-1)
            return out


def compute_normalized_entropy(logits, reduction='mean'):
    """Computes the normalized entorpy for certainty-loss."""
    preds = F.softmax(logits, dim=-1)
    log_preds = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(preds * log_preds, dim=-1)
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)
    return normalized_entropy


class ContinuousThoughtMachine(nn.Module):
    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 memory_length,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 out_dims,
                 memory_hidden_dims,
                 vocab_size,  # Number of possible input tokens (number of distinct characters in FEN string)
                 token_embed_dim,  # Embedding dimension for input tokens
                ):
        super(ContinuousThoughtMachine, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.out_dims = out_dims
        self.memory_length = memory_length
        self.memory_hidden_dims = memory_hidden_dims

        # --- Input Processing  ---
        # Learnable token embeddings
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=token_embed_dim)

        self.attention = nn.MultiheadAttention(self.d_input, heads, batch_first=True)

        #
        self.kv_proj = nn.Sequential(
            nn.Linear(token_embed_dim, d_input),
            nn.LayerNorm(d_input)
        )
        self.q_proj = nn.LazyLinear(self.d_input)

        # --- Core CTM Modules ---
        self.synapses = nn.Sequential(
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        self.trace_processor = nn.Sequential(
            SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model),
            nn.GLU(),
            SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model),
            nn.GLU(),
            Squeeze(-1)
        )

        #  --- Start States ---
        self.register_parameter('start_activated_state', nn.Parameter(
                torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model))),
                requires_grad=True
            ))

        self.register_parameter('start_trace', nn.Parameter(
            torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length))),
            requires_grad=True
        ))

        # --- Synchronisation ---
        self.synch_representation_size_action = (self.n_synch_action * (self.n_synch_action+1))//2
        self.synch_representation_size_out = (self.n_synch_out * (self.n_synch_out+1))//2

        for synch_type, size in [('action', self.synch_representation_size_action), ('out', self.synch_representation_size_out)]:
            print(f"Synch representation size {synch_type}: {size}")

        self.set_synchronisation_parameters('out', self.n_synch_out)
        self.set_synchronisation_parameters('action', self.n_synch_action)

        # --- Output Procesing ---
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

    def set_synchronisation_parameters(self, synch_type: str, n_synch: int):
        left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch)
        synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
        self.register_buffer(f'{synch_type}_neuron_indices_left', left)
        self.register_buffer(f'{synch_type}_neuron_indices_right', right)
        self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch):
        if synch_type == 'out':
            neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
        elif synch_type == 'action':
            neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model)
        return neuron_indices_left, neuron_indices_right

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        if synch_type == 'action':
            n_synch = self.n_synch_action
            selected_left = selected_right = activated_state[:, -n_synch:]
        elif synch_type == 'out':
            n_synch = self.n_synch_out
            selected_left = selected_right = activated_state[:, :n_synch]

        outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
        i, j = torch.triu_indices(n_synch, n_synch)
        pairwise_product = outer[:, i, j]

        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1

        synchronisation = decay_alpha / (torch.sqrt(decay_beta))
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
        embeds = self.embedding(x) # (B, T, embed_dim)
        kv = self.kv_proj(embeds)  # (B, T, d_input)
        return kv

    def compute_certainty(self, current_prediction):
        ne = compute_normalized_entropy(current_prediction)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) # Shape: (B, H, T)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) # Shape: (B, H)

        # --- Storage for outputs per iteration
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)
        all_predictions = []
        all_certainties = []

        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)  # Fix from github user: kuviki
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):
            # --- Calculate Synchronisation for Input Data Interaction ---
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            # --- Interact with Data via Attention ---
            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Activate ---
            activated_state = self.trace_processor(state_trace)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            all_predictions.append(current_prediction)
            all_certainties.append(current_certainty)

            # predictions[..., stepi] = current_prediction
            # certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        predictions = torch.stack(all_predictions, dim=-1) # Stack along a new last dimension
        certainties = torch.stack(all_certainties, dim=-1)

        # --- Return Values ---
        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
        return predictions, certainties, synchronisation_out


def get_loss(predictions, certainties, targets, use_most_certain=True):
    """use_most_certain will select either the most certain point or the final point."""
    losses = nn.CrossEntropyLoss(reduction='none')(
        predictions.float(),
        torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1)
    )

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:, 1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1

    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected) / 2

    return loss, loss_index_2

def calculate_accuracy(predictions, targets, where_most_certain):
    """Calculate the accuracy based on the prediction at the most certain internal tick."""
    B = predictions.size(0)
    device = predictions.device

    predictions_at_most_certain_internal_tick = predictions.argmax(1)[torch.arange(B, device=device), where_most_certain].detach().cpu().numpy()
    accuracy = (targets.detach().cpu().numpy() == predictions_at_most_certain_internal_tick).mean()

    diff = np.abs(targets.detach().cpu().numpy() - predictions_at_most_certain_internal_tick)
    lax_accuracy_3 = (diff <= 1).mean()
    lax_accuracy_5 = (diff <= 2).mean()

    return accuracy, lax_accuracy_3, lax_accuracy_5



PREFETCH_BATCH_SIZE = 8096

# Computes the bucket (class) a certain win probability belongs to.
def get_bucket(win_perc, num_buckets):
    return min(int(win_perc / (1 / num_buckets)), num_buckets - 1)


def iterableFactory(db_path, num_classes):
    assert num_classes > 1, "Bin size should be at least 1"
    def generator():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT padded_ascii_codes, stockfish_win_perc_20 FROM positions"
        )

        while True:
            rows = cursor.fetchmany(PREFETCH_BATCH_SIZE)
            if not rows:
                break
            for row in rows:
                yield dict(x=json.loads(row[0]), y=get_bucket(row[1], num_classes))

        cursor.close()
        conn.close()

    return generator


def prepare_data(num_classes, batch_size=64, train_data_path="train.db", test_data_path="test.db"):
    trainloader = dx.stream_python_iterable(iterableFactory(train_data_path, num_classes)).batch(batch_size)
    testloader = dx.stream_python_iterable(iterableFactory(test_data_path, num_classes)).batch(batch_size)

    return trainloader, testloader


def train(model, trainloader, testloader, iterations, device, lr):
    test_every = 100
    optimizer = torch.optim.AdamW(params=list(model.parameters()), lr=lr, eps=1e-8)
    model.train()

    for stepi in range(iterations):
        batch = next(iter(trainloader))
        inputs, targets = (
            torch.tensor(batch["x"]).to(device),
            torch.tensor(batch["y"]).to(device),
        )
        predictions, certainties, _ = model(inputs, track=False)
        train_loss, where_most_certain = get_loss(predictions, certainties, targets)
        train_accuracy, train_accuracy_3, train_accuracy_5 = calculate_accuracy(predictions, targets, where_most_certain)

        # print(train_loss, train_accuracy, train_accuracy_3, train_accuracy_5)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if stepi % test_every == 0:
            model.eval()
            with torch.inference_mode():
                all_test_predictions = []
                all_test_targets = []
                all_test_where_most_certain = []
                all_test_losses = []
                for i, batch in enumerate(testloader):
                    if i > 100:
                        break
                    inputs, targets = (
                        torch.tensor(batch["x"]).to(device),
                        torch.tensor(batch["y"]).to(device),
                    )
                    predictions, certainties, _ = model(inputs, track=False)
                    test_loss, where_most_certain = get_loss(predictions, certainties, targets)
                    all_test_losses.append(test_loss.item())

                    all_test_predictions.append(predictions)
                    all_test_targets.append(targets)
                    all_test_where_most_certain.append(where_most_certain)

                all_test_predictions = torch.cat(all_test_predictions, dim=0)
                all_test_targets = torch.cat(all_test_targets, dim=0)
                all_test_where_most_certain = torch.cat(all_test_where_most_certain, dim=0)

                test_accuracy, test_accuracy_3, test_accuracy_5 = calculate_accuracy(all_test_predictions, all_test_targets, all_test_where_most_certain)
                test_loss = sum(all_test_losses) / len(all_test_losses)

            wandb.log({
                "train_loss": train_loss.item(),
                "train_accuracy": train_accuracy,
                "train_accuracy_3": train_accuracy_3,
                "train_accuracy_5": train_accuracy_5,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_accuracy_3": test_accuracy_3,
                "test_accuracy_5": test_accuracy_5,
                "step": stepi,
            })
            print(f'Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f} & {train_accuracy_3:.3f} & {train_accuracy_5:.3f} Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f} & {test_accuracy_3:.3f} & {test_accuracy_5:.3f}')
            model.train()

    return model



device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 128
bs=128
lr = 3e-4

trainloader, testloader = prepare_data(num_classes=num_classes, batch_size=bs)

wandb.init(project="CTM-chess", config={
    "iterations": 200000,
    "batch_size": bs,
    "learning_rate": lr,
    "num_classes": num_classes,
    "device": str(device),
    "model": "ContinuousThoughtMachineChess",
})

model = ContinuousThoughtMachine(
    iterations = 50,  # Scale
    d_model = 512,  # Scale
    d_input = 512,  # Scale
    memory_length = 15,  # Scale
    heads = 8,  # scale
    n_synch_out = 128,  # Scale
    n_synch_action = 128,  # Scale
    out_dims=num_classes,
    memory_hidden_dims = 128,
    vocab_size = 128,
    token_embed_dim = 512,
  ).to(device)

batch = next(iter(trainloader))
inputs, targets = (
  torch.tensor(batch["x"]).to(device),
  torch.tensor(batch["y"]).to(device),
)
model(inputs)  # Initialize lazy parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")
model.zero_grad()

model = train(model=model, trainloader=trainloader, testloader=testloader, iterations=200000, device=device, lr=lr)

wandb.finish()
