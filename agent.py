"""
agent.py — Recurrent Actor-Critic Policy Network for End-to-End Bangla HTR (PPO)

Takes state s_t = [h_t; emb(a_{t-1})] combining:
1. Visual feature frame h_t from CRNN backbone
2. Previously predicted token embedding emb(a_{t-1})

Outputs:
1. Actor (Policy distribution pi(a_t | s_t)) over Bangla character vocabulary + <EOS> + <PAD>
2. Critic (State value estimate V(s_t)) for PPO advantage calculation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Dict, Any, Optional


class ActorCriticAgent(nn.Module):
    """Actor-Critic Policy Agent for Sequential Bangla HTR Decoding."""

    def __init__(self, vocab_size: int, feature_dim: int = 512, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_idx = vocab_size - 2  # <EOS> token index
        self.pad_idx = vocab_size - 1  # <PAD> token index

        # Character token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_idx)

        # Recurrent state cell (GRU)
        state_dim = feature_dim + embed_dim
        self.rnn_cell = nn.GRUCell(state_dim, hidden_dim)

        # Actor head (policy distribution pi)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )

        # Critic head (state value function V)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize GRU hidden state with zeros."""
        return torch.zeros(batch_size, self.rnn_cell.hidden_size, device=device)

    def forward(self, feature_frame: torch.Tensor, prev_action: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decoding step forward pass.

        Args:
            feature_frame: (B, feature_dim) visual feature slice h_t
            prev_action: (B,) previously predicted token index a_{t-1}
            hidden: (B, hidden_dim) previous GRU hidden state

        Returns:
            action_logits: (B, vocab_size) policy logits
            state_value: (B, 1) value function baseline V(s_t)
            next_hidden: (B, hidden_dim) updated GRU hidden state
        """
        # Embed previous token
        prev_emb = self.token_embed(prev_action)  # (B, embed_dim)

        # Concatenate visual feature + action embedding
        state_input = torch.cat([feature_frame, prev_emb], dim=1)  # (B, feature_dim + embed_dim)

        # Update recurrent state
        next_hidden = self.rnn_cell(state_input, hidden)  # (B, hidden_dim)

        # Actor logits & Critic value
        action_logits = self.actor(next_hidden)  # (B, vocab_size)
        state_value = self.critic(next_hidden)   # (B, 1)

        return action_logits, state_value, next_hidden

    def sample_action(self, action_logits: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy distribution pi(a_t | s_t).

        Args:
            action_logits: (B, vocab_size)
            deterministic: If True, select argmax token (greedy rollout)

        Returns:
            action: (B,) sampled token index
            log_prob: (B,) log probability log pi(action | s_t)
            entropy: (B,) policy entropy H(pi)
        """
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_actions(self, feature_seq: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates a batch of sequence trajectories for PPO loss computation.

        Args:
            feature_seq: (B, T, feature_dim) visual feature sequence
            actions: (B, T) sequence of actions taken

        Returns:
            log_probs: (B, T) log probabilities of actions
            state_values: (B, T) state values V(s_t)
            entropies: (B, T) policy entropies
        """
        batch_size, seq_len, _ = feature_seq.shape
        device = feature_seq.device

        hidden = self.init_hidden(batch_size, device)
        prev_action = torch.full((batch_size,), self.eos_idx, dtype=torch.long, device=device)

        log_probs_list = []
        state_values_list = []
        entropies_list = []

        for t in range(seq_len):
            h_t = feature_seq[:, t, :]
            target_act = actions[:, t]

            action_logits, state_val, hidden = self.forward(h_t, prev_action, hidden)

            probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(probs)

            log_prob = dist.log_prob(target_act)
            entropy = dist.entropy()

            log_probs_list.append(log_prob.unsqueeze(1))
            state_values_list.append(state_val)
            entropies_list.append(entropy.unsqueeze(1))

            prev_action = target_act

        log_probs = torch.cat(log_probs_list, dim=1)      # (B, T)
        state_values = torch.cat(state_values_list, dim=1) # (B, T, 1) -> squeeze
        entropies = torch.cat(entropies_list, dim=1)        # (B, T)

        return log_probs, state_values.squeeze(-1), entropies
