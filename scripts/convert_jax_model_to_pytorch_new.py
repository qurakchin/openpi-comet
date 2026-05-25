#!/usr/bin/env python3
"""
Convert JAX Pi0/Pi05 checkpoint to new PyTorch format (models_pytorch_new).

Usage:
    python scripts/convert_jax_model_to_pytorch_new.py \
        --checkpoint_dir /mnt/public/zhuchunyang_rl/hf_models/pi05_base \
        --output_path /mnt/public/zhuchunyang_rl/hf_models/pi05_base_pytorch_new \
        --config_name pi05_b1k-base

Reference: openpi/examples/convert_jax_model_to_pytorch.py
"""

import json
import os
import pathlib
import shutil

import numpy as np
import torch
import tyro

import openpi.models.model as _model
import openpi.training.config as _config


def load_jax_params(checkpoint_dir: str) -> dict:
    """Load JAX checkpoint parameters."""
    params = _model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype="float32"
    )
    return params


def convert_siglip(params: dict) -> dict:
    """Convert SigLIP ViT parameters from JAX to new PyTorch format."""
    pt = {}
    img = params["PaliGemma"]["img"]

    # Patch embedding (Conv2d): JAX (H, W, C_in, C_out) -> PT (C_out, C_in, H, W)
    pt["img.stem.weight"] = torch.from_numpy(
        img["embedding"]["kernel"].transpose(3, 2, 0, 1)
    )
    pt["img.stem.bias"] = torch.from_numpy(img["embedding"]["bias"])

    # Encoder blocks
    eb = img["Transformer"]["encoderblock"]

    ln0_scale = eb["LayerNorm_0"]["scale"]  # (27, 1152)
    ln0_bias = eb["LayerNorm_0"]["bias"]
    ln1_scale = eb["LayerNorm_1"]["scale"]
    ln1_bias = eb["LayerNorm_1"]["bias"]

    # MLP
    dense0_kernel = eb["MlpBlock_0"]["Dense_0"]["kernel"]  # (27, 1152, 4304)
    dense0_bias = eb["MlpBlock_0"]["Dense_0"]["bias"]      # (27, 4304)
    dense1_kernel = eb["MlpBlock_0"]["Dense_1"]["kernel"]  # (27, 4304, 1152)
    dense1_bias = eb["MlpBlock_0"]["Dense_1"]["bias"]      # (27, 1152)

    # Attention Q, K, V, O
    mha = eb["MultiHeadDotProductAttention_0"]
    q_kernel = mha["query"]["kernel"]   # (27, 1152, 16, 72)
    q_bias = mha["query"]["bias"]       # (27, 16, 72)
    k_kernel = mha["key"]["kernel"]     # (27, 1152, 16, 72)
    k_bias = mha["key"]["bias"]         # (27, 16, 72)
    v_kernel = mha["value"]["kernel"]   # (27, 1152, 16, 72)
    v_bias = mha["value"]["bias"]       # (27, 16, 72)
    o_kernel = mha["out"]["kernel"]     # (27, 16, 72, 1152)
    o_bias = mha["out"]["bias"]         # (27, 1152)

    width = 1152  # So400m/14 width

    for i in range(27):
        prefix = f"img.encoder.layers.{i}"

        # LayerNorm
        pt[f"{prefix}.norm1.weight"] = torch.from_numpy(ln0_scale[i])
        pt[f"{prefix}.norm1.bias"] = torch.from_numpy(ln0_bias[i])
        pt[f"{prefix}.norm2.weight"] = torch.from_numpy(ln1_scale[i])
        pt[f"{prefix}.norm2.bias"] = torch.from_numpy(ln1_bias[i])

        # Attention: JAX has separate Q, K, V -> PT MultiheadAttention has concatenated in_proj
        # JAX Dense: x @ kernel (kernel shape: in_features x out_features)
        # PT Linear: x @ weight.T (weight shape: out_features x in_features)
        # For equivalence: weight = kernel.T
        q_w = torch.from_numpy(q_kernel[i].reshape(width, width).T)
        k_w = torch.from_numpy(k_kernel[i].reshape(width, width).T)
        v_w = torch.from_numpy(v_kernel[i].reshape(width, width).T)
        q_b = torch.from_numpy(q_bias[i].reshape(width))
        k_b = torch.from_numpy(k_bias[i].reshape(width))
        v_b = torch.from_numpy(v_bias[i].reshape(width))

        pt[f"{prefix}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        pt[f"{prefix}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        # Out projection: JAX (16, 72, 1152) -> PT (1152, 1152)
        pt[f"{prefix}.attn.out_proj.weight"] = torch.from_numpy(
            o_kernel[i].reshape(width, width).T
        )
        pt[f"{prefix}.attn.out_proj.bias"] = torch.from_numpy(o_bias[i])

        # MLP
        pt[f"{prefix}.mlp.fc1.weight"] = torch.from_numpy(dense0_kernel[i].T)
        pt[f"{prefix}.mlp.fc1.bias"] = torch.from_numpy(dense0_bias[i])
        pt[f"{prefix}.mlp.fc2.weight"] = torch.from_numpy(dense1_kernel[i].T)
        pt[f"{prefix}.mlp.fc2.bias"] = torch.from_numpy(dense1_bias[i])

    # Encoder final norm
    pt["img.encoder.norm.weight"] = torch.from_numpy(
        img["Transformer"]["encoder_norm"]["scale"]
    )
    pt["img.encoder.norm.bias"] = torch.from_numpy(
        img["Transformer"]["encoder_norm"]["bias"]
    )

    # Head projection: JAX (1152, 2048) -> PT Linear (2048, 1152)
    pt["img.head.weight"] = torch.from_numpy(img["head"]["kernel"].T)
    pt["img.head.bias"] = torch.from_numpy(img["head"]["bias"])

    # Position embedding — JAX stores as learned parameter, not computed dynamically
    pt["img.pos_embedding"] = torch.from_numpy(img["pos_embedding"])  # (1, 256, 1152)

    return pt


def convert_llm(params: dict, pi05: bool) -> dict:
    """Convert LLM (Gemma) parameters from JAX to new PyTorch format.

    Converts both experts:
      - Expert 0: PaliGemma (width=2048)
      - Expert 1: Action Expert (width=1024)
    """
    pt = {}
    llm = params["PaliGemma"]["llm"]

    # Embedder
    pt["llm.embedder.embedding.weight"] = torch.from_numpy(
        llm["embedder"]["input_embedding"]
    )

    layers_data = llm["layers"]
    paligemma_width = 2048
    action_width = 1024

    # Expert 0 (PaliGemma)
    q_einsum = layers_data["attn"]["q_einsum"]["w"]         # (18, 8, 2048, 256)
    kv_einsum = layers_data["attn"]["kv_einsum"]["w"]       # (18, 2, 1, 2048, 256)
    o_einsum = layers_data["attn"]["attn_vec_einsum"]["w"]  # (18, 8, 256, 2048)

    mlp_gating = layers_data["mlp"]["gating_einsum"]  # (18, 2, 2048, 16384)
    mlp_linear = layers_data["mlp"]["linear"]          # (18, 16384, 2048)

    pre_attn_scale = layers_data["pre_attention_norm"]["scale"]  # (18, 2048)
    pre_ffw_scale = layers_data["pre_ffw_norm"]["scale"]          # (18, 2048)

    # Expert 1 (Action Expert)
    q_einsum_1 = layers_data["attn"]["q_einsum_1"]["w"]         # (18, 8, 1024, 256)
    kv_einsum_1 = layers_data["attn"]["kv_einsum_1"]["w"]       # (18, 2, 1, 1024, 256)
    o_einsum_1 = layers_data["attn"]["attn_vec_einsum_1"]["w"]  # (18, 8, 256, 1024)

    mlp_gating_1 = layers_data["mlp_1"]["gating_einsum"]  # (18, 2, 1024, 4096)
    mlp_linear_1 = layers_data["mlp_1"]["linear"]          # (18, 4096, 1024)

    n_layers = q_einsum.shape[0]  # 18

    for i in range(n_layers):
        # --- Expert 0: PaliGemma ---
        # Q projection: JAX (8, 2048, 256) -> PT (num_heads*head_dim, width) = (2048, 2048)
        q_w0 = torch.from_numpy(
            q_einsum[i].transpose(0, 2, 1).reshape(paligemma_width, paligemma_width)
        )
        pt[f"llm.layers.{i}.attn.q_proj.0.weight"] = q_w0

        # K projection: JAX kv_einsum[i, 0, 0]: (2048, 256) -> PT (256, 2048)
        k_w0 = torch.from_numpy(kv_einsum[i, 0, 0].T)
        pt[f"llm.layers.{i}.attn.k_proj.0.weight"] = k_w0

        # V projection: JAX kv_einsum[i, 1, 0]: (2048, 256) -> PT (256, 2048)
        v_w0 = torch.from_numpy(kv_einsum[i, 1, 0].T)
        pt[f"llm.layers.{i}.attn.v_proj.0.weight"] = v_w0

        # O projection: JAX (8, 256, 2048) -> PT (width, num_heads*head_dim) = (2048, 2048)
        o_w0 = torch.from_numpy(
            o_einsum[i].reshape(paligemma_width, paligemma_width).T
        )
        pt[f"llm.layers.{i}.attn.o_proj.0.weight"] = o_w0

        # Norms: regular RMSNorm
        pt[f"llm.layers.{i}.pre_attention_norms.0.scale"] = torch.from_numpy(pre_attn_scale[i])
        pt[f"llm.layers.{i}.pre_ffw_norms.0.scale"] = torch.from_numpy(pre_ffw_scale[i])

        # FFN
        pt[f"llm.layers.{i}.mlps.0.w_gating"] = torch.from_numpy(mlp_gating[i])
        pt[f"llm.layers.{i}.mlps.0.w_linear"] = torch.from_numpy(mlp_linear[i])

        # --- Expert 1: Action Expert ---
        # Q projection: JAX (8, 1024, 256) -> PT (2048, 1024)
        q_w1 = torch.from_numpy(
            q_einsum_1[i].transpose(0, 2, 1).reshape(paligemma_width, action_width)
        )
        pt[f"llm.layers.{i}.attn.q_proj.1.weight"] = q_w1

        # K projection: JAX kv_einsum_1[i, 0, 0]: (1024, 256) -> PT (256, 1024)
        k_w1 = torch.from_numpy(kv_einsum_1[i, 0, 0].T)
        pt[f"llm.layers.{i}.attn.k_proj.1.weight"] = k_w1

        # V projection
        v_w1 = torch.from_numpy(kv_einsum_1[i, 1, 0].T)
        pt[f"llm.layers.{i}.attn.v_proj.1.weight"] = v_w1

        # O projection: JAX (8, 256, 1024) -> PT (width, num_heads*head_dim) = (1024, 2048)
        o_w1 = torch.from_numpy(
            o_einsum_1[i].reshape(paligemma_width, action_width).T
        )
        pt[f"llm.layers.{i}.attn.o_proj.1.weight"] = o_w1

        # FFN
        pt[f"llm.layers.{i}.mlps.1.w_gating"] = torch.from_numpy(mlp_gating_1[i])
        pt[f"llm.layers.{i}.mlps.1.w_linear"] = torch.from_numpy(mlp_linear_1[i])

        # Norms: adaptive for expert 1 when pi05
        if pi05:
            pre_attn_1 = layers_data["pre_attention_norm_1"]
            pre_ffw_1 = layers_data["pre_ffw_norm_1"]

            # Dense_0/kernel: (18, 1024, 3072) -> PT (3072, 1024)
            pt[f"llm.layers.{i}.pre_attention_norms.1.ada_modulation.weight"] = torch.from_numpy(
                pre_attn_1["Dense_0"]["kernel"][i].T
            )
            pt[f"llm.layers.{i}.pre_attention_norms.1.ada_modulation.bias"] = torch.from_numpy(
                pre_attn_1["Dense_0"]["bias"][i]
            )
            pt[f"llm.layers.{i}.pre_ffw_norms.1.ada_modulation.weight"] = torch.from_numpy(
                pre_ffw_1["Dense_0"]["kernel"][i].T
            )
            pt[f"llm.layers.{i}.pre_ffw_norms.1.ada_modulation.bias"] = torch.from_numpy(
                pre_ffw_1["Dense_0"]["bias"][i]
            )
        else:
            pt[f"llm.layers.{i}.pre_attention_norms.1.scale"] = torch.from_numpy(
                layers_data["pre_attention_norm_1"]["scale"][i]
            )
            pt[f"llm.layers.{i}.pre_ffw_norms.1.scale"] = torch.from_numpy(
                layers_data["pre_ffw_norm_1"]["scale"][i]
            )

    # Final norms
    pt["llm.final_norms.0.scale"] = torch.from_numpy(llm["final_norm"]["scale"])

    if pi05:
        final_norm_1 = llm["final_norm_1"]
        pt["llm.final_norms.1.ada_modulation.weight"] = torch.from_numpy(
            final_norm_1["Dense_0"]["kernel"].T
        )
        pt["llm.final_norms.1.ada_modulation.bias"] = torch.from_numpy(
            final_norm_1["Dense_0"]["bias"]
        )
    else:
        pt["llm.final_norms.1.scale"] = torch.from_numpy(llm["final_norm_1"]["scale"])

    return pt


def convert_projections(params: dict, pi05: bool) -> dict:
    """Convert action/time projection parameters."""
    pt = {}
    proj = params

    if pi05:
        proj_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    else:
        proj_keys = [
            "state_proj", "action_in_proj", "action_out_proj",
            "action_time_mlp_in", "action_time_mlp_out",
        ]

    for key in proj_keys:
        if key not in proj:
            continue
        kernel = proj[key]["kernel"]
        bias = proj[key]["bias"]
        if isinstance(kernel, dict):
            kernel = kernel["value"]
            bias = bias["value"]

        pt[f"{key}.weight"] = torch.from_numpy(np.array(kernel).T)
        pt[f"{key}.bias"] = torch.from_numpy(np.array(bias))

    return pt


def convert_checkpoint(
    checkpoint_dir: str,
    output_path: str,
    config_name: str,
    precision: str = "bfloat16",
):
    """Full conversion pipeline."""
    print(f"Loading JAX checkpoint from: {checkpoint_dir}")
    params = load_jax_params(checkpoint_dir)

    # Load config to get model parameters
    model_config = _config.get_config(config_name).model
    pi05 = model_config.pi05
    print(f"Config: {config_name}, pi05={pi05}")

    # Convert each component
    print("Converting SigLIP ViT...")
    siglip_pt = convert_siglip(params)

    print("Converting LLM...")
    llm_pt = convert_llm(params, pi05)

    print("Converting projections...")
    proj_pt = convert_projections(params, pi05)

    # Merge all parameters
    all_pt = {}
    for d in [siglip_pt, llm_pt, proj_pt]:
        for k, v in d.items():
            all_pt[k] = v.contiguous() if not v.is_contiguous() else v

    print(f"Total converted parameters: {len(all_pt)}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save as safetensors
    import safetensors.torch
    safetensors.torch.save_file(all_pt, os.path.join(output_path, "model.safetensors"))

    # Copy assets
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # Save config for reference
    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "max_token_len": model_config.max_token_len,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "pi05": model_config.pi05,
        "pcd": model_config.pcd,
        "dtype": model_config.dtype,
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Conversion complete! Saved to {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str,
    precision: str = "bfloat16",
):
    convert_checkpoint(
        checkpoint_dir=checkpoint_dir,
        output_path=output_path,
        config_name=config_name,
        precision=precision,
    )


if __name__ == "__main__":
    tyro.cli(main)
