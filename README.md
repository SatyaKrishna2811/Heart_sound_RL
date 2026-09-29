# Heart_sound_RL

**RL-guided adaptive temporal–spectral feature fusion for murmur detection in heart sounds.**

Phonocardiogram classifiers usually hard-code their representation: one wavelet, one frequency
band, one window length, one fusion rule. This project lets a reinforcement-learning agent choose
it per recording from label-free signal descriptors. The agent picks the wavelet family, the
decomposition level, the frequency band, the temporal window and which of three branches to fuse
(temporal / STFT / wavelet-packet). An attention-fusion CNN then detects murmurs. Everything runs
on **CirCor DigiScope v1.0.3** (PhysioNet) and lives in Jupyter notebooks.

It combines two papers:

- **Xiao & Wang, PLOS ONE 2025** (RL wavelet-base selection for ECG), transferred to PCG
  and generalised from "which wavelet" to the full representation.
- **MST-ResFLNet, IEEE 2025** (multiscale temporal–spectral fusion for heart sounds): the
  pyramid-spectrum stem, SE attention, temporal attention pooling and focal loss used in every branch.

It also covers two courses: **24AIM304 Reinforcement Learning** (bandits, MDPs, DP, Monte
Carlo, TD, n-step, Dyna, prioritized sweeping, MCTS) and **24AIM301 Signal & Image Processing**
(IIR/FIR, PSD, coherence, cepstrum, homomorphic filtering, wavelets, PCA/ICA, image segmentation,
texture, compression).

> Status: `main` holds the mid-semester half (`notebooks/midsem/`, tag `midsem-submission`), complete and executed. The end-semester half is developed on the [`endsem`](https://github.com/SatyaKrishna2811/Heart_sound_RL/tree/endsem) branch.

## How it works

```
PCG ─► QC ─► resample 2 kHz ─► Butterworth 20–800 Hz ─► spike removal ─► robust norm
        │
        ├─► label-free descriptors ─► PCA ─► k-means ─► context k            (RL state)
        │
        ├─► agent π(k) = (wavelet, level, band, window, fusion)               (RL action, 2,100 options)
        │
        ├─► T: raw window ─► 1-D CNN ─┐
        ├─► S: STFT rows in band ─► 2-D CNN ─┼─► attention fusion ─► p(murmur)
        └─► W: wavelet packet rows in band ─► 2-D CNN ─┘
                                                     │
reward = .40 Δbalanced-acc + .25 Δsens + .20 Δspec + .10 robustness − .05 complexity
         (per recording, out-of-fold, paid by a weight-sharing "supernet")
```

The configuration choice is a 5-step finite MDP (wavelet → level → band → window → fusion).
It is solved with DP (the planning optimum), Monte Carlo, SARSA / Expected SARSA / Q-learning /
Double Q / n-step SARSA, Dyna-Q, prioritized sweeping, rollouts and MCTS, with PPO as a deep-RL
comparison. Rewards come from a supernet: one fusion network trained under random configurations,
whose out-of-fold predictions score all 2,100 configurations for every recording.

## Mid-semester results

Classical reward oracle, patient-independent validation split (450 recordings), 3 seeds where applicable.

**The representation matters.** On validation, the DP-optimal representation policy beats the
fixed baseline configuration (db6, level 5, 20–800 Hz, 2 s, all branches):

| policy | macro-F1 | balanced acc. | AUROC |
|---|---|---|---|
| fixed baseline configuration | 0.606 | 0.637 | 0.727 |
| DP optimum, 8 signal contexts | **0.665** | **0.697** | **0.794** |

**Choosing the wavelet alone barely matters.** RLWBS-style bandits, CV-WBS and EE-WBS all land within
±0.01 validation macro-F1 of a fixed db6. The gain comes from choosing band, window and fusion.

**RL finds good policies with 1.4 % of the exhaustive search budget** (60,000 reward queries vs 4.41 M):

| agent | fraction of the DP-optimal value | val macro-F1 |
|---|---|---|
| Monte Carlo control | **0.87** | 0.652 |
| MCTS (UCT) | 0.78 | 0.639 |
| rollout algorithm | 0.77 | 0.643 |
| n-step SARSA (n = 3) | 0.46 | 0.660 |
| Q-learning | 0.39 | 0.647 |
| SARSA / Expected SARSA | 0.32 | 0.630 / 0.638 |
| Dyna-Q (n = 20) / prioritized sweeping | 0.40 / 0.41 | 0.621 / 0.640 |
| PPO (continuous state, MultiDiscrete action) | — | 0.671 |

With sparse terminal rewards, very noisy per-recording rewards and 16,800 (context, configuration)
leaves, methods that use the actual return (MC, rollouts, MCTS) beat bootstrapping methods at this
budget. Longer n-step returns help steadily, and Q-learning keeps improving with more episodes
(0.33 → 0.55 at 4× the budget). On a synthetic MDP with a known optimum, every learner reaches
93–100 % of the optimal value (notebook 10), so the gap comes from the data, not the implementation.

**Signal processing.** Cepstral heart rate from the homomorphic envelope is within 10 bpm of the
annotated rate for 67 % of recordings. 100–200 Hz relative band power is the most murmur-separable
single band (AUROC 0.62). FastICA recovers a PCG source from a two-channel mixture with |ρ| = 1.00.

## Project structure: mid-semester and end-semester

| | Mid-semester | End-semester |
|---|---|---|
| notebooks | [`notebooks/midsem/`](notebooks/midsem) 01–12 | [`notebooks/endsem/`](notebooks/endsem) 13–22 |
| results | [`results/midsem/`](results/midsem) | [`results/endsem/`](results/endsem) |
| question | Does the representation matter, and can tabular RL find a good one with far fewer evaluations than exhaustive search? | Does the RL-chosen representation make a deep fusion network better, more robust and cheaper than any fixed choice? |
| reward oracle | classical (out-of-fold logistic regression) | weight-sharing supernet (the deep network family itself) |
| course coverage | 24AIM301 Units 1–3 · 24AIM304 Units 1–4 | 24AIM301 Unit 4 · 24AIM304 CO5 (real-world application) |
| git tag | `midsem-submission` | `endsem-submission` |

```
notebooks/
  00_project_plan.ipynb                   refined plan: problem, papers, course mapping, MDP, changes vs draft
  common/                                 shared library notebooks, loaded with %run
    00_config.ipynb                       every hyper-parameter, the action space, reward weights, use_half()
    01_signal_lib.ipynb                   CirCor parsing, preprocessing, descriptors, representations, metrics
    02_rl_lib.ipynb                       action space, reward oracle, contexts, bandits, DP/MC/TD/planning
    03_deep_lib.ipynb                     MST-ResFLNet-style branches, attention fusion, supernet, training
  midsem/
    01_download_dataset                   parallel, resumable, SHA-256-verified mirror of CirCor v1.0.3
    02_metadata_and_patient_split         recording-level murmur labels, 70/15/15 patient split
    03_quality_control_and_preprocessing  QC, resampling, Butterworth band-pass, spike removal, RL-state descriptors
    04_signal_processing_analysis         IIR vs FIR, PSD, cepstrum, homomorphic envelope, coherence, PCA, ICA, TF-image processing
    05_candidate_representations_and_features   wavelet packets, STFT, feature groups for 2,100 configurations
    06_classical_reward_oracle            out-of-fold logistic-regression oracle (1,244 models)
    07_rl_state_contexts                  PCA + k-means signal contexts, choice of K
    08_bandit_wavelet_selection           ε-greedy, UCB, gradient bandit, Thompson vs CV-WBS / EE-WBS
    09_mdp_dynamic_programming            value iteration, policy iteration
    10_monte_carlo_and_td_control         MC, SARSA, Expected SARSA, Q-learning, Double Q, n-step SARSA
    11_planning_dyna_prioritized_sweeping_mcts   Dyna-Q, prioritized sweeping, rollout, MCTS/UCT
    12_ppo_deep_rl_comparison             PPO with a MultiDiscrete action
  endsem/
    13_deep_reward_oracle_supernet        one network trained under random configurations scores all 2,100
    14_rl_agents_on_deep_oracle           every agent re-run on the deep oracle
    15_policy_selection_and_rl_ablations  agent chosen on validation; policy-level ablations
    16_deep_baselines                     raw CNN, STFT CNN, db6 wavelet CNN, fixed fusions, supernet control
    17_rl_guided_fusion_models            RL wavelet-only, full RL, supernet fine-tuning, joint policy refinement
    18_deep_ablations                     one ingredient removed at a time
    19_final_evaluation                   test split: recording & patient level, bootstrap CI, Unknown murmurs, outcome
    20_robustness_and_compression         white / pink / impulse noise, JPEG-style DCT coding of TF images
    21_explainability                     selected actions, branch attention, temporal attention vs cardiac phase
    22_results_summary                    all deliverables in one place
data/      raw/ interim/ processed/  (git-ignored, regenerated by the notebooks)
models/    training histories, predictions and summaries (weights git-ignored)
docs/      the original draft plan (PDF)
```

Code cells carry no comments. The explanations are in the markdown cells.

## Quickstart

```bash
git clone https://github.com/SatyaKrishna2811/Heart_sound_RL && cd Heart_sound_RL
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name pcg_rl
jupyter lab notebooks/
```

Run `midsem/01` … `midsem/12`, then `endsem/13` … `endsem/22`. `midsem/01_download_dataset`
mirrors CirCor v1.0.3 (about 560 MB, 10,434 files, all SHA-256 verified) into
`data/raw/circor_v1.0.3/`. It is the equivalent of the official command:

```bash
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.3/
```

Hardware used: one RTX 4050 laptop GPU (6 GB) and 22 CPU threads. The mid-semester half runs on
the CPU in about 15 minutes after the download. The end-semester half needs about 4 GPU hours.

## Data and citation

All experiments use **CirCor DigiScope v1.0.3 only** (3,163 recordings, 942 patients; the public
portion of the collection). License ODC-By 1.0.

> Oliveira J, Renna F, Costa P, et al. *The CirCor DigiScope Dataset: From Murmur Detection to Murmur Classification.* IEEE JBHI 26(6):2524–2535, 2022. https://physionet.org/content/circor-heart-sound/1.0.3/
>
> Xiao Q, Wang C. *Adaptive wavelet base selection for deep learning-based ECG diagnosis: A reinforcement learning approach.* PLOS ONE 20(2): e0318070, 2025.
>
> *MST-ResFLNet: Multiscale Temporal–Spectral Feature Fusion for Heart Sound Signal Analysis.* IEEE, 2025. https://ieeexplore.ieee.org/document/11301917
