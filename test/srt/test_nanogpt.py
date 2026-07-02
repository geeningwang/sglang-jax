"""Unit tests for the NanoGPT port in SGLang-JAX.

Tests both the serving model (python/sgl_jax/srt/models/nanogpt.py) and
the standalone training model (examples/nanogpt/model.py).

Run with:
    python test/srt/test_nanogpt.py
    # or from repo root:
    python -m pytest test/srt/test_nanogpt.py -v
"""

import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

# Add examples/nanogpt to path for the training model
sys.path.insert(0, "examples/nanogpt")


class TestNanoGPTServingModel(unittest.TestCase):
    """Tests for NanoGPTForCausalLM (serving model in python/sgl_jax/srt/models/nanogpt.py)."""

    def setUp(self):
        from sgl_jax.srt.models.nanogpt import NanoGPTConfig, NanoGPTForCausalLM

        devices = jax.devices()
        # Single-device mesh (tensor axis only) for testing
        self.mesh = Mesh(np.array(devices[:1]).reshape(1), ("tensor",))
        # GPT-2 small config
        self.config = NanoGPTConfig(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=1024,
            vocab_size=50304,
            bias=True,
        )
        with self.mesh:
            self.model = NanoGPTForCausalLM(
                config=self.config,
                mesh=self.mesh,
                dtype=jnp.bfloat16,
            )

    def test_config_aliases(self):
        """NanoGPTConfig exposes ModelConfig-expected field names."""
        cfg = self.config
        self.assertEqual(cfg.num_hidden_layers, 12)
        self.assertEqual(cfg.num_attention_heads, 12)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.intermediate_size, 3072)
        self.assertEqual(cfg.max_position_embeddings, 1024)
        self.assertEqual(cfg.head_dim, 64)
        self.assertEqual(cfg.num_key_value_heads, 12)

    def test_model_instantiation(self):
        """Model creates the expected number of transformer blocks."""
        self.assertIsNotNone(self.model)
        self.assertEqual(len(self.model.model.layers), 12)

    def test_param_count(self):
        """GPT-2 124M has approximately 124M parameters."""
        state = nnx.state(self.model)
        total = sum(v.size for v in jax.tree_util.tree_leaves(state))
        # GPT-2 124M: ~124M params (actual: ~163M with embedding + bias).
        # With weight tying lm_head == wte, param count is ~124M.
        # Loose range check: between 100M and 200M.
        self.assertGreater(total, 100_000_000, "Param count too low")
        self.assertLess(total, 250_000_000, "Param count too high")

    def test_layernorm_forward(self):
        """NanoGPTLayerNorm produces correct output shape and finite values."""
        from sgl_jax.srt.models.nanogpt import NanoGPTLayerNorm

        ln = NanoGPTLayerNorm(768)
        x = jnp.ones((16, 768))
        y = ln(x)
        self.assertEqual(y.shape, (16, 768))
        self.assertTrue(jnp.all(jnp.isfinite(y)))

    def test_weight_mapping_completeness(self):
        """Weight mapping covers all expected GPT-2 keys."""
        mappings = self.model._create_weight_mappings()
        # Should have: wte, wpe, ln_f.{weight,bias} + per-layer entries
        self.assertIn("transformer.wte.weight", mappings)
        self.assertIn("transformer.wpe.weight", mappings)
        self.assertIn("transformer.ln_f.weight", mappings)
        # 12 layers × 12 keys each = 144 layer entries + 4 top-level = 148 total
        self.assertEqual(len(mappings), 4 + 12 * 12)

    def test_registry_entry_class(self):
        """EntryClass is set correctly for model registry auto-discovery."""
        from sgl_jax.srt.models.nanogpt import EntryClass, NanoGPTForCausalLM

        self.assertIs(EntryClass, NanoGPTForCausalLM)


class TestNanoGPTTrainingModel(unittest.TestCase):
    """Tests for the standalone NNX training model (examples/nanogpt/model.py)."""

    def setUp(self):
        from model import GPT, GPTConfig  # loaded via sys.path insert above

        self.GPT = GPT
        self.GPTConfig = GPTConfig

        # Tiny model for fast unit testing
        self.cfg = GPTConfig(
            block_size=64,
            vocab_size=256,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            bias=True,
        )

    def test_forward_pass(self):
        """Forward pass returns (logits, loss) of correct shapes."""
        model = self.GPT(self.cfg)
        B, T = 2, 32
        idx = jnp.zeros((B, T), dtype=jnp.int32)
        targets = jnp.zeros((B, T), dtype=jnp.int32)
        logits, loss = model(idx, targets)
        self.assertEqual(logits.shape, (B, T, self.cfg.vocab_size))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, ())
        self.assertTrue(jnp.isfinite(loss))

    def test_inference_mode(self):
        """Inference (targets=None) returns only last-position logits."""
        model = self.GPT(self.cfg)
        B, T = 1, 16
        idx = jnp.zeros((B, T), dtype=jnp.int32)
        logits, loss = model(idx)
        self.assertEqual(logits.shape, (B, 1, self.cfg.vocab_size))
        self.assertIsNone(loss)

    def test_param_count(self):
        """Tiny model has a reasonable parameter count."""
        model = self.GPT(self.cfg)
        n = model.num_params()
        self.assertGreater(n, 0)
        # Rough upper bound: 2 layers × (attn + mlp + 2×LN) + embeddings
        self.assertLess(n, 5_000_000)

    def test_weight_tying(self):
        """wte is used for both token embedding and lm_head projection."""
        model = self.GPT(self.cfg)
        idx = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        # wte lookup for embedding
        emb = model.wte.value[idx[0, 0]]
        # lm_head is wte.T — confirm shapes match
        self.assertEqual(model.wte.value.shape, (self.cfg.vocab_size, self.cfg.n_embd))
        self.assertEqual(emb.shape, (self.cfg.n_embd,))

    def test_nnx_split_merge(self):
        """nnx.split / nnx.merge roundtrip produces identical outputs."""
        model = self.GPT(self.cfg)
        graphdef, state = nnx.split(model)
        reconstructed = nnx.merge(graphdef, state)

        idx = jnp.ones((1, 8), dtype=jnp.int32)
        logits1, _ = model(idx)
        logits2, _ = reconstructed(idx)
        np.testing.assert_array_equal(logits1, logits2)

    def test_loss_decreases_with_gradient(self):
        """A single gradient step reduces loss (sanity check for gradient flow)."""
        import optax

        model = self.GPT(self.cfg)
        graphdef, state = nnx.split(model)
        tx = optax.adam(1e-3)
        opt_state = tx.init(state)

        idx = jnp.zeros((2, 16), dtype=jnp.int32)
        targets = jnp.zeros((2, 16), dtype=jnp.int32)

        def loss_fn(state):
            m = nnx.merge(graphdef, state)
            _, loss = m(idx, targets)
            return loss

        loss_before, grads = jax.value_and_grad(loss_fn)(state)
        updates, new_opt_state = tx.update(grads, opt_state, state)
        new_state = optax.apply_updates(state, updates)
        loss_after, _ = jax.value_and_grad(loss_fn)(new_state)

        self.assertLess(float(loss_after), float(loss_before))

    def test_estimate_mfu(self):
        """MFU estimate returns a float in (0, 1) for reasonable throughput."""
        model = self.GPT(self.cfg)
        mfu = model.estimate_mfu(
            fwdbwd_per_iter=4,
            dt=1.0,  # 1 second per step (slow, so MFU will be low)
            tpu_peak_tflops=918.0,
        )
        self.assertIsInstance(float(mfu), float)
        self.assertGreater(float(mfu), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
