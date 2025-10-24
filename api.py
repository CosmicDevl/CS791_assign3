#!/usr/bin/env python3
import pickle
import math
from typing import Dict, List

class FastRewardCalculator:
    def __init__(self, cache_file: str, epsilon: float = 1e-9):
        """
        cache_file: pickle with at least
         - 'trigram_probs': Dict[str, float], key = "tok1,tok2,tok3", value = P(t3|t1,t2)
        """
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        self._tri_probs: Dict[str, float] = cache["trigram_probs"]
        self._eps: float = float(epsilon)

        # Expose a token LM object with .logp expected by SMC/TSMC code.
        # Keep naming stable: reward_calc.token_lm.logp(...)
        self.token_lm = _TokenLM(self._tri_probs, self._eps)

    def calculate_reward_tokens(self, tokens: List[str], normalize: bool = True) -> float:
        """
        Calculates the reward for a sequence of tokens based on trigram rarity.

        The reward R(x) for a sequence is defined as the sum of the
        negative log-probabilities of its constituent trigrams:
        R(x) = sum_{t=3..T} -log(P(token_t | token_{t-2}, token_{t-1}))

        Args:
            tokens (List[str]):
                List of token strings.

            normalize (bool, optional):
                Whether to compute the average reward per trigram (True)
                or the unnormalized total reward (False). The unnormalized
                reward corresponds to R(x^(i)) in the metric definition.

        Returns:
            float:
                The calculated reward. Returns 0.0 if fewer than 3 tokens
                are provided (as no trigrams can be formed).
        """
        # A sequence must have at least 3 tokens to form a trigram.
        if len(tokens) < 3:
            return 0.0

        total_neg_log_p = 0.0
        num_trigrams = 0

        # Iterate over all trigrams in the sequence
        # range(2, len(tokens)) ensures we start from the third token (index 2)
        for i in range(2, len(tokens)):
            t1 = tokens[i-2]
            t2 = tokens[i-1]
            t3 = tokens[i]
            
            # Get log P(t3 | t1, t2)
            log_p = self.token_lm.logp(t1, t2, t3)
            
            # Reward is the negative log-probability
            # R(x) = sum(-log_p)
            total_neg_log_p -= log_p
            num_trigrams += 1

        if normalize:
            # Return the average reward per trigram
            # num_trigrams will be > 0 if len(tokens) >= 3
            return total_neg_log_p / num_trigrams
        else:
            # Return the total unnormalized reward for the sequence
            return total_neg_log_p


class _TokenLM:
    """Minimal token-trigram LM with logp only. Internal use."""
    def __init__(self, tri_probs: Dict[str, float], eps: float):
        self._tri = tri_probs
        self._eps = eps

    @staticmethod
    def _key(t1: str, t2: str, t3: str) -> str:
        return f"{t1},{t2},{t3}"

    def logp(self, t1: str, t2: str, t3: str) -> float:
        """Return log P(t3 | t1, t2) with epsilon floor."""
        p = self._tri.get(self._key(t1, t2, t3), 0.0)
        if p <= 0.0:
            p = self._eps
        return math.log(p)
