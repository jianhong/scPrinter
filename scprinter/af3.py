from __future__ import annotations

import gc
import os

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
import copy
import itertools
import json
import math
import pickle
import random
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import pyranges
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec
from scipy.sparse import csc_matrix, vstack
from scipy.stats import zscore
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm, trange

from . import TFBS, footprint, motifs
from .datasets import (
    FigR_motifs_human_meme,
    FigR_motifs_mouse_meme,
    pretrained_seq_TFBS_model0,
    pretrained_seq_TFBS_model1,
)
from .genome import Genome
from .io import _get_group_atac, _get_group_atac_bw, get_bias_insertions, scPrinter
from .motifs import Motifs
from .preprocessing import export_bigwigs
from .seq import dataloader, interpretation
from .utils import *


def modisco_composite_report(
    modisco_h5: str | Path | list[Path],
    save_path: str | Path,
    meme_motif: os.PathLike | Literal["human", "mouse"],
    motif_prefix: str | list[str] = "",
    delta_effect_path: str | Path | list[Path] = None,
    qval_threshold=1e-3,
    trim_threshold=0.3,
    n_top_hits=1,
    selected_patterns: list[str] | list[list[str]] | None = None,
):
    """
    Create the modisco report that contains the motif logos, the motif matches, the delta effects.

    Parameters
    ----------
    modisco_h5: str | Path | list[Path]
        The path to the modisco h5 file(s)
    save_path: str | Path
        The path to save the modisco report
    meme_motif: os.PathLike | Literal ['human', 'mouse']
        The path to a motif database in meme format or use the default human/mouse FigR motif database
    delta_effect_path: str | Path
        The path to the delta effects file. This should be your save_path argument passed to `scp.tl.delta_effects_seq2print`
        When provided as a list, make it the same length as the modisco_h5
    motif_prefix: str
        The prefix for the denovo motifs if any. This should be the same as the prefix argument passed to `scp.tl.delta_effects_seq2print`.
        When provided as a list, make it the same length as the modisco_h5
    is_writing_tomtom_matrix: bool
        Whether to write the tomtom matrix
    top_n_matches: int
        The number of top matches to keep
    trim_threshold: float
        The threshold for trimming the motifs
    trim_min_length: int
        The minimum length for trimming the motifs
    selected_patterns: str | list[str] | None
        The selected patterns to plot. If None, all patterns will be plotted. If a list of strings, only the patterns in the list will be plotted. e.g. ['pos_patterns.pattern_60', 'pos_patterns.pattern_4', 'pos_patterns.pattern_2', 'pos_patterns.pattern_27']
    Returns
    -------


    """
    output_dir = os.path.dirname(save_path)
    save_name = os.path.basename(save_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if type(meme_motif) is str:
        if meme_motif == "human":
            meme_motif = (FigR_motifs_human_meme,)
        elif meme_motif == "mouse":
            meme_motif = FigR_motifs_mouse_meme

    if type(modisco_h5) is not list:
        modisco_h5 = [modisco_h5]
    if type(delta_effect_path) is not list:
        delta_effect_path = [delta_effect_path]
    if type(motif_prefix) is str:
        motif_prefix = [motif_prefix]
    if selected_patterns is None:
        selected_patterns = [None] * len(modisco_h5)
    elif type(selected_patterns[0]) is str:
        selected_patterns = [selected_patterns]
    img_path_suffixs = [os.path.join(output_dir, f"{h5}_figs") for h5 in modisco_h5]
    for img_path_suffix in img_path_suffixs:
        if not os.path.exists(img_path_suffix):
            os.makedirs(img_path_suffix)
    return interpretation.modisco_report.report_composite_motif(
        modisco_h5,
        output_dir,
        save_name,
        img_path_suffixs,
        meme_motif,
        delta_effect_path,
        motif_prefix,
        qval_threshold,
        trim_threshold,
        n_top_hits,
        selected_patterns,
    )


def composite_af3(
    composite_df_rows,
    modelSeeds=None,
    species="human",
    singleton_jobs=False,
    json_output=None,
):
    # The modelSeeds can be any integer between 0 and 4,294,967,295
    # The maximum value for modelSeeds
    MAX_SEED_VALUE = 4_294_967_295

    # Check if modelSeeds is None
    if modelSeeds is None:
        # Assign a random integer in the valid range
        modelSeeds = random.randint(0, MAX_SEED_VALUE)

    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    json_dict = []
    existing_jobs_names = set()
    for i, composite_df_row in composite_df_rows.iterrows():
        job_name = composite_df_row["pattern"]
        job_name = job_name.replace(".", "_")
        job_name = job_name.replace(" ", "_")
        if species == "human":
            species_term = "Homo sapiens"
        elif species == "mouse":
            species_term = "Mus musculus"
        else:
            print("Species not in human or mouse, make sure to provide sth like Homo sapiens.")
            species_term = species

        tf1 = composite_df_row["Target_ID1"]
        tf2 = composite_df_row["Target_ID2"]
        protein_seqs = [
            get_protein_sequence(tf1, species_term),
            get_protein_sequence(tf2, species_term),
        ]
        dna_seq = composite_df_row["consensus_seq"]

        # Generate the reverse complement of the DNA sequence
        dna_reverse = "".join(complement[base] for base in dna_seq[::-1]) if dna_seq else None

        # Template generation
        sequences = []

        # Add protein sequences to the template
        for seq in protein_seqs:
            sequences.append({"proteinChain": {"sequence": seq, "count": 1}})

        # Add DNA sequence and its reverse complement if provided
        dna_seq_info = [
            {"dnaSequence": {"sequence": dna_seq, "count": 1}},
            {"dnaSequence": {"sequence": dna_reverse, "count": 1}},
        ]
        sequences += dna_seq_info

        # Generate the JSON payload
        json_dict.append(
            {"name": f"{job_name}_{tf1}_{tf2}", "modelSeeds": [modelSeeds], "sequences": sequences}
        )
        existing_jobs_names.add(f"{job_name}_{tf1}_{tf2}")

        if singleton_jobs:
            for i, seq in enumerate(protein_seqs):
                tf = tf1 if i == 0 else tf2
                if f"{job_name}_{tf}" in existing_jobs_names:
                    continue
                existing_jobs_names.add(f"{job_name}_{tf}")

                sequences = [{"proteinChain": {"sequence": seq, "count": 1}}] + dna_seq_info
                json_dict.append(
                    {"name": f"{job_name}_{tf}", "modelSeeds": [modelSeeds], "sequences": sequences}
                )

    if json_output:
        with open(json_output, "w") as f:
            json.dump(json_dict, f, indent=4)
    else:
        return json_dict


def extract_af3(file_paths, smooth_plddt=None):
    """
    Extracts plDDT scores for each chain from the AlphaFold JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary where keys are chain IDs and values are lists of plDDT scores.
    """
    # Load the JSON file
    if type(file_paths) is not list:
        file_paths = [file_paths]
    final_plddt_by_chain = {}
    final_contact_probs = []
    final_contact_probs_by_chain = {}
    final_paes = []
    final_paes_by_chain = {}
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Initialize a dictionary to store plDDT scores by chain
        plddt_by_chain = defaultdict(list)

        # Get chain IDs and plDDT scores
        chain_ids = data["atom_chain_ids"]
        plddts = data["atom_plddts"]
        contact_probs = np.array(data["contact_probs"])
        token_chain_ids = data["token_chain_ids"]
        contact_probs = (contact_probs + contact_probs.T) / 2
        paes = np.array(data["pae"])
        paes = (paes + paes.T) / 2

        # Combine chain IDs and plDDT scores
        for chain, plddt in zip(chain_ids, plddts):
            plddt_by_chain[chain].append(plddt)

        plddt_by_chain = dict(plddt_by_chain)
        plddt_by_chain = {x: np.array(plddt_by_chain[x]) for x in plddt_by_chain}

        for chain in plddt_by_chain:
            plddt = plddt_by_chain[chain]
            if smooth_plddt > 0:
                plddt = rz_conv(np.array(plddt), smooth_plddt) / (2 * smooth_plddt)
            if chain not in final_plddt_by_chain:
                final_plddt_by_chain[chain] = []
                final_contact_probs_by_chain[chain] = []
                final_paes_by_chain[chain] = []

            final_plddt_by_chain[chain].append(plddt)

        chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        start = 0

        chain_intervals = {}
        start = 0

        for i in range(1, len(token_chain_ids)):
            if token_chain_ids[i] != token_chain_ids[start]:
                # Store the interval for the current chain
                chain_intervals[token_chain_ids[start]] = [start, i]
                start = i  # Update the start for the next chain

        # Handle the last chain
        chain_intervals[token_chain_ids[start]] = [start, len(token_chain_ids)]

        for chain in chain_intervals:
            start, end = chain_intervals[chain]
            final_contact_probs_by_chain[chain].append(contact_probs[start:end, start:end])
            final_paes_by_chain[chain].append(paes[start:end, start:end])

        final_contact_probs.append(contact_probs)
        final_paes.append(paes)
    for chain in final_plddt_by_chain:
        final_plddt_by_chain[chain] = np.array(final_plddt_by_chain[chain])
        final_contact_probs_by_chain[chain] = np.array(final_contact_probs_by_chain[chain])
        final_paes_by_chain[chain] = np.array(final_paes_by_chain[chain])
    return {
        "plddt": final_plddt_by_chain,
        "contact_probs": final_contact_probs,
        "contact_probs_by_chain": final_contact_probs_by_chain,
        "paes": final_paes,
        "paes_by_chain": final_paes_by_chain,
        "chain_intervals": chain_intervals,
    }


def composite_analysis(composite_paths, tf1_paths=None, tf2_paths=None, smooth_plddt=None):
    co = extract_af3(composite_paths, smooth_plddt)
    if tf1_paths is not None:
        tf1 = extract_af3(tf1_paths, smooth_plddt)
    else:
        tf1 = None
    if tf2_paths is not None:
        tf2 = extract_af3(tf2_paths, smooth_plddt)
    else:
        tf2 = None
    chain_intervals = co["chain_intervals"]

    A_start, A_end = chain_intervals["A"]
    B_start, B_end = chain_intervals["B"]
    C_start, C_end = chain_intervals["C"]

    A = np.mean(co["contact_probs"], axis=0)[A_start:A_end, B_start:B_end]
    B = np.mean(co["contact_probs"], axis=0)[A_start:A_end, C_start:C_end]
    C = np.mean(co["contact_probs"], axis=0)[B_start:B_end, C_start:C_end]

    # Create a figure with GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(
        2, 2, width_ratios=[10, 1], height_ratios=[10, 1], figure=fig, wspace=0.1, hspace=0.1
    )

    # Define a common colormap and value range
    cmap = "Reds"
    vmin = 0.0
    vmax = 0.1

    # Heatmap for A
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(A, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title("Contact of TF1 - TF2")
    ax1.xaxis.set_visible(False)

    # Heatmap for B (aligned to rows of A)
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    im2 = ax2.imshow(B, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title("Contact of TF1 - DNA")
    ax2.tick_params(labelleft=False)  # Remove y-axis labels to align visually

    # Heatmap for C (aligned to columns of A)
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    im3 = ax3.imshow(C.T, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax3.set_title("Contact of TF2 - DNA")
    ax3.tick_params(labeltop=False)  # Align x-axis with A

    # Add a single colorbar below Matrix B
    cbar_ax = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation="horizontal")
    # plt.tight_layout()

    A = np.mean(co["paes"], axis=0)[A_start:A_end, B_start:B_end]
    B = np.mean(co["paes"], axis=0)[A_start:A_end, C_start:C_end]
    C = np.mean(co["paes"], axis=0)[B_start:B_end, C_start:C_end]

    # Create a figure with GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(
        2, 2, width_ratios=[10, 1], height_ratios=[10, 1], figure=fig, wspace=0.1, hspace=0.1
    )

    # Define a common colormap and value range
    cmap = "RdBu"
    vmin = 0.0
    vmax = 30

    # Heatmap for A
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(A, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title("PAE of TF1 - TF2")
    ax1.xaxis.set_visible(False)  # Hide x-axis labels
    # ax1.yaxis.set_visible(False)  # Hide y-axis labels

    # Heatmap for B (aligned to rows of A)
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    im2 = ax2.imshow(B, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title("PAE of TF1 - DNA")
    ax2.tick_params(labelleft=False)  # Remove y-axis labels to align visually

    # Heatmap for C (aligned to columns of A)
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    im3 = ax3.imshow(C.T, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax3.set_title("PAE of TF2 - DNA")
    ax3.tick_params(labeltop=False)  # Align x-axis with A

    # Add a single colorbar below Matrix B
    cbar_ax = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation="horizontal")
    # plt.tight_layout()

    if tf1 is not None:

        plt.figure(figsize=(15, 5))
        sns.lineplot(
            x=np.stack(
                [np.arange(co["plddt"]["A"].shape[1])] * co["plddt"]["A"].shape[0], axis=0
            ).reshape((-1)),
            y=co["plddt"]["A"].reshape((-1)),
            alpha=0.75,
            label="co-fold tf1",
            linewidth=1,
        )
        sns.lineplot(
            x=np.stack(
                [np.arange(tf1["plddt"]["A"].shape[1])] * tf1["plddt"]["A"].shape[0], axis=0
            ).reshape((-1)),
            y=tf1["plddt"]["A"].reshape((-1)),
            alpha=0.75,
            label="tf1",
            linewidth=1,
        )
        plt.title("Composite - TF1 plddt")
    if tf2 is not None:
        plt.figure(figsize=(15, 5))
        sns.lineplot(
            x=np.stack(
                [np.arange(co["plddt"]["B"].shape[1])] * co["plddt"]["B"].shape[0], axis=0
            ).reshape((-1)),
            y=co["plddt"]["B"].reshape((-1)),
            alpha=0.75,
            label="co-fold tf2",
            linewidth=1,
        )
        sns.lineplot(
            x=np.stack(
                [np.arange(tf2["plddt"]["A"].shape[1])] * tf2["plddt"]["A"].shape[0], axis=0
            ).reshape((-1)),
            y=tf2["plddt"]["A"].reshape((-1)),
            alpha=0.75,
            label="tf2",
            linewidth=1,
        )
        plt.title("Composite - TF2 plddt")
