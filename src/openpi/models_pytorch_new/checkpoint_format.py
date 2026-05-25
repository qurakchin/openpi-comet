"""Checkpoint format conversion — module-level patching.

Each module class (RMSNorm, Attention, FeedForward, etc.) owns its
old-format conversion logic.  This module provides:

- ``set_old_ckpt_format(enable)`` — global toggle
- ``_is_old_format(state_dict)`` — detect old-format keys
- ``pi0_old_state_dict(model)`` — build old-format state dict from a Pi0
- ``pi0_load_old_state_dict(model, old_sd)`` — load old-format params into Pi0

Pi0.<load_state_dict/state_dict> overrides call these helpers so that
``safetensors.torch.{load_model,save_model}`` work transparently.
"""

from __future__ import annotations

import torch


def old_to_new_state_dict(old_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert old-format flat state dict to new-format.

    Handles key renaming and weight transformations:
      - SigLIP Q/K/V concat → in_proj_weight/bias
      - LLM MLP gate/up transpose+stack → w_gating (2, features, hidden_dim)
      - LLM MLP down transpose → w_linear
    """
    new_sd: dict[str, torch.Tensor] = {}

    # --- Helper: build a SigLIP encoder prefix pair ---
    _SIGLIP_OLD = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."

    # Stem
    for suf in (".weight", ".bias"):
        ok = _SIGLIP_OLD + "embeddings.patch_embedding" + suf
        if ok in old_sd:
            new_sd["img.stem" + suf] = old_sd[ok]

    # Pos embedding
    ok = _SIGLIP_OLD + "embeddings.position_embedding.weight"
    if ok in old_sd:
        new_sd["img.pos_embedding"] = old_sd[ok]

    # Encoder layers (0..26)
    for i in range(27):
        op = f"{_SIGLIP_OLD}encoder.layers.{i}."
        np = f"img.encoder.layers.{i}."

        # Norms
        for old_n, new_n in [("layer_norm1", "norm1"), ("layer_norm2", "norm2")]:
            for suf in (".weight", ".bias"):
                ok = f"{op}{old_n}{suf}"
                if ok in old_sd:
                    new_sd[f"{np}{new_n}{suf}"] = old_sd[ok]

        # Attention: QKV concat
        qkv_w = []
        qkv_b = []
        for proj in ("q_proj", "k_proj", "v_proj"):
            wk = f"{op}self_attn.{proj}.weight"
            bk = f"{op}self_attn.{proj}.bias"
            if wk in old_sd:
                qkv_w.append(old_sd[wk])
            if bk in old_sd:
                qkv_b.append(old_sd[bk])
        if qkv_w:
            new_sd[f"{np}attn.in_proj_weight"] = torch.cat(qkv_w, dim=0)
        if qkv_b:
            new_sd[f"{np}attn.in_proj_bias"] = torch.cat(qkv_b, dim=0)

        # Out proj
        for suf in (".weight", ".bias"):
            ok = f"{op}self_attn.out_proj{suf}"
            if ok in old_sd:
                new_sd[f"{np}attn.out_proj{suf}"] = old_sd[ok]

        # MLP
        for name in ("fc1", "fc2"):
            for suf in (".weight", ".bias"):
                ok = f"{op}mlp.{name}{suf}"
                if ok in old_sd:
                    new_sd[f"{np}mlp.{name}{suf}"] = old_sd[ok]

    # Post layernorm
    for suf in (".weight", ".bias"):
        ok = _SIGLIP_OLD + "post_layernorm" + suf
        if ok in old_sd:
            new_sd["img.encoder.norm" + suf] = old_sd[ok]

    # Multi-modal projector
    for suf in (".weight", ".bias"):
        ok = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear" + suf
        if ok in old_sd:
            new_sd["img.head" + suf] = old_sd[ok]

    # --- PaliGemma LLM (expert 0) ---
    _PALI_LLM = "paligemma_with_expert.paligemma.model.language_model."
    for i in range(18):
        op = f"{_PALI_LLM}layers.{i}."
        np = f"llm.layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            ok = f"{op}self_attn.{proj}.weight"
            if ok in old_sd:
                new_sd[f"{np}attn.{proj}.0.weight"] = old_sd[ok]

        # MLP: gate + up → w_gating (transpose + stack)
        gk = f"{op}mlp.gate_proj.weight"
        uk = f"{op}mlp.up_proj.weight"
        if gk in old_sd and uk in old_sd:
            gate_t = old_sd[gk].T.contiguous()  # (hidden_dim, features) → (features, hidden_dim)
            up_t = old_sd[uk].T.contiguous()
            new_sd[f"{np}mlps.0.w_gating"] = torch.stack([gate_t, up_t], dim=0)

        dk = f"{op}mlp.down_proj.weight"
        if dk in old_sd:
            new_sd[f"{np}mlps.0.w_linear"] = old_sd[dk].T.contiguous()

        # Norms
        for old_n, new_n in [("input_layernorm", "pre_attention_norms"), ("post_attention_layernorm", "pre_ffw_norms")]:
            ok = f"{op}{old_n}.weight"
            if ok in old_sd:
                new_sd[f"{np}{new_n}.0.scale"] = old_sd[ok]

    # Final norm expert 0
    ok = _PALI_LLM + "norm.weight"
    if ok in old_sd:
        new_sd["llm.final_norms.0.scale"] = old_sd[ok]

    # --- Gemma Action Expert (expert 1) ---
    _GEMMA_EXP = "paligemma_with_expert.gemma_expert.model."
    for i in range(18):
        op = f"{_GEMMA_EXP}layers.{i}."
        np = f"llm.layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            ok = f"{op}self_attn.{proj}.weight"
            if ok in old_sd:
                new_sd[f"{np}attn.{proj}.1.weight"] = old_sd[ok]

        gk = f"{op}mlp.gate_proj.weight"
        uk = f"{op}mlp.up_proj.weight"
        if gk in old_sd and uk in old_sd:
            gate_t = old_sd[gk].T.contiguous()
            up_t = old_sd[uk].T.contiguous()
            new_sd[f"{np}mlps.1.w_gating"] = torch.stack([gate_t, up_t], dim=0)

        dk = f"{op}mlp.down_proj.weight"
        if dk in old_sd:
            new_sd[f"{np}mlps.1.w_linear"] = old_sd[dk].T.contiguous()

        # Norms (ada_modulation)
        for old_n, new_n in [("input_layernorm", "pre_attention_norms"), ("post_attention_layernorm", "pre_ffw_norms")]:
            for suf in (".weight", ".bias"):
                ok = f"{op}{old_n}.dense{suf}"
                if ok in old_sd:
                    new_sd[f"{np}{new_n}.1.ada_modulation{suf}"] = old_sd[ok]

    # Final norm expert 1
    for suf in (".weight", ".bias"):
        ok = _GEMMA_EXP + "norm.dense" + suf
        if ok in old_sd:
            new_sd["llm.final_norms.1.ada_modulation" + suf] = old_sd[ok]

    # --- lm_head → embedder (tied weights — either key works) ---
    lm_head_key = None
    if "paligemma_with_expert.gemma_expert.lm_head.weight" in old_sd:
        lm_head_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
    elif "paligemma_with_expert.paligemma.lm_head.weight" in old_sd:
        lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    if lm_head_key is not None:
        new_sd["llm.embedder.embedding.weight"] = old_sd[lm_head_key]

    # --- Action head (same names in both formats) ---
    for k in old_sd:
        if k.startswith(
            ("action_in_proj", "action_out_proj", "time_mlp_", "state_proj", "action_time_mlp_", "pointnet.")
        ):
            new_sd[k] = old_sd[k]

    return new_sd


def new_to_old_state_dict(new_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert new-format flat state dict to old-format."""
    old_sd: dict[str, torch.Tensor] = {}

    _SIGLIP_OLD = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."

    # Stem
    for suf in (".weight", ".bias"):
        nk = "img.stem" + suf
        if nk in new_sd:
            old_sd[_SIGLIP_OLD + "embeddings.patch_embedding" + suf] = new_sd[nk]

    if "img.pos_embedding" in new_sd:
        old_sd[_SIGLIP_OLD + "embeddings.position_embedding.weight"] = new_sd["img.pos_embedding"]

    # Encoder layers
    for i in range(27):
        op = f"{_SIGLIP_OLD}encoder.layers.{i}."
        np = f"img.encoder.layers.{i}."

        for old_n, new_n in [("layer_norm1", "norm1"), ("layer_norm2", "norm2")]:
            for suf in (".weight", ".bias"):
                nk = f"{np}{new_n}{suf}"
                if nk in new_sd:
                    old_sd[f"{op}{old_n}{suf}"] = new_sd[nk]

        # QKV split
        wk = f"{np}attn.in_proj_weight"
        if wk in new_sd:
            q_w, k_w, v_w = torch.chunk(new_sd[wk], 3, dim=0)
            old_sd[f"{op}self_attn.q_proj.weight"] = q_w.contiguous()
            old_sd[f"{op}self_attn.k_proj.weight"] = k_w.contiguous()
            old_sd[f"{op}self_attn.v_proj.weight"] = v_w.contiguous()
        bk = f"{np}attn.in_proj_bias"
        if bk in new_sd:
            q_b, k_b, v_b = torch.chunk(new_sd[bk], 3, dim=0)
            old_sd[f"{op}self_attn.q_proj.bias"] = q_b.contiguous()
            old_sd[f"{op}self_attn.k_proj.bias"] = k_b.contiguous()
            old_sd[f"{op}self_attn.v_proj.bias"] = v_b.contiguous()

        for suf in (".weight", ".bias"):
            nk = f"{np}attn.out_proj{suf}"
            if nk in new_sd:
                old_sd[f"{op}self_attn.out_proj{suf}"] = new_sd[nk]

        for name in ("fc1", "fc2"):
            for suf in (".weight", ".bias"):
                nk = f"{np}mlp.{name}{suf}"
                if nk in new_sd:
                    old_sd[f"{op}mlp.{name}{suf}"] = new_sd[nk]

    for suf in (".weight", ".bias"):
        nk = "img.encoder.norm" + suf
        if nk in new_sd:
            old_sd[_SIGLIP_OLD + "post_layernorm" + suf] = new_sd[nk]

    for suf in (".weight", ".bias"):
        nk = "img.head" + suf
        if nk in new_sd:
            old_sd["paligemma_with_expert.paligemma.model.multi_modal_projector.linear" + suf] = new_sd[nk]

    # --- PaliGemma LLM ---
    _PALI_LLM = "paligemma_with_expert.paligemma.model.language_model."
    for i in range(18):
        op = f"{_PALI_LLM}layers.{i}."
        np = f"llm.layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            nk = f"{np}attn.{proj}.0.weight"
            if nk in new_sd:
                old_sd[f"{op}self_attn.{proj}.weight"] = new_sd[nk]

        gk = f"{np}mlps.0.w_gating"
        if gk in new_sd:
            w = new_sd[gk]  # (2, features, hidden_dim)
            old_sd[f"{op}mlp.gate_proj.weight"] = w[0].T.contiguous()
            old_sd[f"{op}mlp.up_proj.weight"] = w[1].T.contiguous()

        nk = f"{np}mlps.0.w_linear"
        if nk in new_sd:
            old_sd[f"{op}mlp.down_proj.weight"] = new_sd[nk].T.contiguous()

        for old_n, new_n in [("input_layernorm", "pre_attention_norms"), ("post_attention_layernorm", "pre_ffw_norms")]:
            nk = f"{np}{new_n}.0.scale"
            if nk in new_sd:
                old_sd[f"{op}{old_n}.weight"] = new_sd[nk]

    nk = "llm.final_norms.0.scale"
    if nk in new_sd:
        old_sd[_PALI_LLM + "norm.weight"] = new_sd[nk]

    # --- Gemma Expert ---
    _GEMMA_EXP = "paligemma_with_expert.gemma_expert.model."
    for i in range(18):
        op = f"{_GEMMA_EXP}layers.{i}."
        np = f"llm.layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            nk = f"{np}attn.{proj}.1.weight"
            if nk in new_sd:
                old_sd[f"{op}self_attn.{proj}.weight"] = new_sd[nk]

        gk = f"{np}mlps.1.w_gating"
        if gk in new_sd:
            w = new_sd[gk]  # (2, features, hidden_dim)
            old_sd[f"{op}mlp.gate_proj.weight"] = w[0].T.contiguous()
            old_sd[f"{op}mlp.up_proj.weight"] = w[1].T.contiguous()

        nk = f"{np}mlps.1.w_linear"
        if nk in new_sd:
            old_sd[f"{op}mlp.down_proj.weight"] = new_sd[nk].T.contiguous()

        for old_n, new_n in [("input_layernorm", "pre_attention_norms"), ("post_attention_layernorm", "pre_ffw_norms")]:
            for suf in (".weight", ".bias"):
                nk = f"{np}{new_n}.1.ada_modulation{suf}"
                if nk in new_sd:
                    old_sd[f"{op}{old_n}.dense{suf}"] = new_sd[nk]

    for suf in (".weight", ".bias"):
        nk = "llm.final_norms.1.ada_modulation" + suf
        if nk in new_sd:
            old_sd[_GEMMA_EXP + "norm.dense" + suf] = new_sd[nk]

    # --- embedder → lm_head ---
    if "llm.embedder.embedding.weight" in new_sd:
        old_sd["paligemma_with_expert.paligemma.lm_head.weight"] = new_sd["llm.embedder.embedding.weight"]
        old_sd["paligemma_with_expert.gemma_expert.lm_head.weight"] = new_sd["llm.embedder.embedding.weight"]

    # --- Action head (pass through) ---
    for k in new_sd:
        if k.startswith(
            ("action_in_proj", "action_out_proj", "time_mlp_", "state_proj", "action_time_mlp_", "pointnet.")
        ):
            old_sd[k] = new_sd[k]

    return old_sd
