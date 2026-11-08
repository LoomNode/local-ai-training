> Compiled by a research subagent on 2026-10-04 for the speed round (see `notes/SPEED.md`), at Kyle's request to re-review the source material. Not checked against its sources line by line; what it marks unverified is unverified.

# Fly-style learning and cheap training: literature review (2026-10-04)

Method: web only; numbers read from the opened source, not search snippets. [n] maps to the URL list at the end. "Unverified" = could not open or confirm. bpb = compressed bytes x 8 / input bytes (my arithmetic).

## Answer in brief

- Fly learning is cheap because only one layer learns (KC-MBON synapses), locally, in one to a few trials, on a frozen sparse expansion, touching only the ~5% of synapses that were active [3][5]. Consolidation is neither cheap nor local [13][15][16].
- No fly-copying design here shows cheaper LM pretraining than a tuned backprop baseline at matched quality. FlyVec's cheapness claim uses unmatched baselines [18]; wiring-versus-control tests are null [22][23].
- Backprop-free training at LM scale is not cheaper [53][57][59]; SOLO (2026) saves pipeline memory but adds FLOPs [58].
- What has evidence: deterministic hashed lookup memories [44][46][47], chunkwise fast-weight kernels [32][33], modestly frozen random sub-layers [56]. Retrieval shifts cost to inference and storage [38][39].

## Part 1 - Biology

- **Fixed:** PN-to-KC wiring. About 50 glomerular channels expand to about 2,000 KCs, ratio 40 [4]. Each KC has 2-11 claws (mean 7, n=200) from an apparently random PN mix [1]. EM gives about 6 PNs per KC [8] and shows food-responsive PN types over-converging, so wiring is not exactly random [10]. Odor-dopamine pairing does not change KC responses [3].
- **Plastic:** KC-MBON synapses. 15 compartments tile the lobes, with 21 MBON types and 20 dopamine neuron (DAN) types [2].
- **Rule:** one pairing of a 1 s odor with four 1 ms pulses of PPL1-gamma1pedc DAN activity cut the paired-odor MBON response from 118 to 24 spikes for at least 40 min (unpaired odor 110 to 83). Reversed order did nothing; MBON spikes were not required [3]. The write needs KC activity, dopamine and order, not postsynaptic firing. Cross-odor depression follows KC overlap (30-33% overlap, small spill-over) [3]. DopR1 and DopR2 couple to different second messengers and direct depression or potentiation by timing [14]; one DAN type can write, reduce, or write the opposite sign [13].
- **Trials and compartments:** one pairing gives short-term memory in gamma1pedc, largely gone by 24 h. PPL1-alpha3 barely learns from one pairing, but spaced repetition gives 1-day and 4-day memory. Stability is "an intrinsic property of the MB compartment"; a fast-decay aversive and a slow-decay appetitive trace for one odor flip valence over a day [13]; three DAN pathways give memories of different stability [17].
- **Sparseness and APL:** KC sparseness arises from fast EPSP decay, low PN convergence and high thresholds [11]. The GABAergic APL neuron closes a KC-APL-KC loop; disrupting it lowered sparseness, raised inter-odor correlation and blocked discrimination of similar (not dissimilar) odors [12]. About 5% of KCs fire per odor [5]; a novelty compartment uses ~20 KCs, 5-10% of 350 [6]. Theory: small KC in-degree maximises dimension, and with plastic input synapses dense wiring wins, favouring frozen input plus plastic output [4].
- **Consolidation:** long-term memory needs spaced training; whether short-term converts to long-term or forms in parallel is open [13]. It is metabolically gated: mushroom-body energy-flux upregulation is necessary and sufficient [15], and a 2026 paper shows spaced training re-arms fructose-sensing neurons so post-learning sugar triggers consolidation [16]. FlyWire vs hemibrain: MB excitation varies, E/I ratio held [9].
- **Limits:** the fly learns odor valences, not sequences [7]. I found no per-trial energy measurement (unverified).

## Part 2 - Machine learning

### 2.1 Fly-derived

| Work | Mechanism; trained vs frozen | Cost evidence | Quality vs baseline; scale |
|---|---|---|---|
| FlyHash [5] | Frozen sparse binary projection (each KC samples ~6 of 50 inputs) + top-k; nothing trained | Projection 20x fewer ops than dense Gaussian at equal retrieval; at equal ops top-k beats random selection (SIFT k=4 mAP 32.4% vs 17.7%) | Similarity search only; d up to 784 |
| BioHash [20] | Same expansion, but learned by a local Hebbian rule | Not reported | MNIST mAP@All, k=4: 49.3 vs FlyHash 20.0, LSH 13.8; k=16: 54.9 vs 26.3, 20.3 |
| FlyVec [18] | KC layer (K=400-600) learned by winner-take-all Hebbian rule on binary context-word vectors | 17 min/epoch on 3 V100, 76 min on 44 CPU cores; 2-3 epochs enough. Own GloVe run 10 h on 16 threads; BERT time quoted from a 250B-token paper. Unfair: hardware, corpus; no energy | WiC 57.7 vs BERT-large 61.2; SST-2 77.1 vs GloVe 78.3, BERT 90.8. Embedder, not LM; 6B tokens |
| Krotov-Hopfield [19] | Local competitive Hebbian first layer (2,000 units), supervised top | Authors: "main drawback ... is that it is slow" | Test error MNIST 1.52% vs 1.5% SGD; CIFAR-10 49.25% vs 44.74% |
| FlyModel [21] | Sparse expansion; only active-KC to target synapses update; one pass | No compute numbers | MNIST-20, 5 tasks: 0.86 vs 0.77 (BI-R), 0.58 (EWC); 0.11 below offline net at 20 classes |
| Flies Are All You Need [22] | Frozen 1.17B LM + frozen MaleCNS reservoir (166,700 nodes); 278,528-parameter readout | None reported (laptop) | NLL 1.381995 to 1.359816; same-size direct-input control 1.359328 (better); 1,236 tokens; authors: advantage "not demonstrated" |
| FlyGPT [23] (self-reported) | 5,000-neuron MaleCNS subgraph RNN, edges trained | n/a | Val 1.609 vs 1.613 nats/char for degree-preserving scramble; ~578k-param GRU 1.522, transformer 1.548, fly 1.587 |
| fly-llm-hf, gpt-fly [24] (model cards) | Frozen fly reservoir + 52.8M readout; GPT-2-small with frozen fly masks | n/a | Permuted wiring 4.27 vs real 3.84 val; masked GPT-2 1.82 vs dense 1.14 |

Connectome reservoirs resist overfitting on chaotic time series [25]; BrainTrace is linear-memory online learning for a fly spiking model [26]; neither is language.

### 2.2 Sparse associative memory and fast weights

- Attention approximates Kanerva's sparse distributed memory (fixed random addresses, summed patterns) under stated conditions [27]; Kanerva 1988 itself not opened (unverified). A top-k SDMLP trained by gradient descent reached 71% Split-CIFAR10 (10K neurons) vs FlyModel 82%, with 2,000 epochs per split [28]. Modern Hopfield layers [29]: no training-cost evidence found.
- Linear attention is a fast-weight programmer; exact storage is capped by key dimension, and a parameter-free expansion (DPFP) raises it, much like a frozen KC layer (my analogy) [31]. Its 63K vs 33K words/s speedup is against plain-PyTorch softmax. Ba's fast weights converge faster than LSTMs on a synthetic retrieval task, with an S-step inner loop per step [30].
- Fast-weight training cost is a kernel question. FlashLinearAttention beats FlashAttention-2 as a layer at 1K [32]. At 1.3B on an H100, GLA beats Mamba and DeltaNet is "close to GLA" but lags [33]; Gated DeltaNet trails Mamba2 by 2-3K tokens/s and Transformer++ is best at 2K [34] (figure values not extracted). Test-time-training's dual form is over 5x faster than naive yet "can still require substantial wall-clock time" [35]. Sparse routed slot writes reuse GLA kernels [36].

### 2.3 Memory instead of weights

- Continuous cache, no training: WikiText-2 99.3 to 68.9, WikiText-103 48.7 to 40.8; gains shrink with training data [37].
- kNN-LM: WikiText-103 18.65 to 16.12 with no training; a 100M-token model plus 3B-token datastore scores 13.73 vs 15.17 for training on 3B [38]. Cost moves to inference: 3,847 to 277 tokens/s (10-30x slower), 200 GB for 103M tokens; efficiency tricks recover up to 6x [39].
- infini-gram: index 7 bytes/token, 1.4T tokens in ~2 days on one 128-core node; Pile perplexity GPT-Neo-125M 13.50 to 10.76 (-22%), Llama-2-70B -11% [40].
- RETRO: 7.5B matches GPT-3 with 25x fewer parameters, not 25x less compute [41]. A reproduction costs 12-26% more GPU-hours; RETRO 148M scores 12.99 in 1,560 h vs GPT 410M 13.18 in 3,600 h, 43% of the hours (my arithmetic) [42]. RETRO exploits eval leakage more than baselines [41]; leakage control in [42] not stated (unverified).
- Memorizing Transformers: 8K memory matches a 5x larger model; step time 0.2 s to 0.25 s (8K) and 0.6 s (65K) at ~200M parameters [43].
- Deterministic hashed address, learned table: fixed random-hash routing beat learned Switch (23.23 vs 23.73 test ppl) [44]; Engram at 27B, 262B tokens, iso-FLOPs and iso-parameters: MMLU +3.4, BBH +5.0, a 100B table offloaded for under 3% overhead [46]. In the NanoGPT speedrun (GPT-2-small to 3.28 loss in 0.665 min on 8 H100 vs 45 min for llm.c) the "Bigram Hash Embedding" record moved 1.748 to 1.655 min (-5.3%) [47].

### 2.4 Online context mixers (enwik8, 100 MB)

lstm-compress (online LSTM) 1.63 at 92 us/byte; paq8hp12 1.30 at 37.7 us/byte (63 min); NNCP v3.2 1.19 at 242 us/byte; cmix v21 1.17, 18.1 h and 21 GB on CPU [48][49]. NNCP v2 (187M transformer, RTX 3090) is 1.20 bpb at 1.94 kB/s, about 14 h per 100 MB (derived) [50]. cmix on Silesia's Samba source tar: 0.59 bpb [49][51]. These are whole-file adaptive averages, not held-out scores [50]. cmix mixes hashed-context predictions in a gated linear network: "Every neuron in the network directly tries to minimize cross entropy, so there is no backpropagation of gradients between layers" [49]. That is sparse address, local update, one pass, at high per-byte cost.

### 2.5 Training without full backprop

- DFA, 6-layer Transformer, WikiText-103: BP 29.8 ppl vs DFA 52.0 (block-wise) or 93.3 (sub-layer) [52]. At 57M-509M parameters and 30B tokens DFA is never more compute-efficient, even credited 2ND FLOPs against BP's 6ND [53]. Forward-Forward is "somewhat slower", with no language results [54]; target-prop and feedback-alignment fail on CIFAR/ImageNet [55].
- 2026: forward-gradient on a 16M GPT-style model gives ppl 387 vs BP 150 [59]. EGGROLL int8 recurrent LM: 3.40 vs 3.58 bpb for a backprop transformer at equal data batch, but a 2^20 population costs about 180x the GPU-hours; its "100x" is versus naive evolution strategies [57]. SOLO (28 Sep 2026 preprint): 340M-2B, perplexity gap 3.1% (two modules) to 6.1% (four) at 2B, +11-13% FLOPs; its 1.44x is versus pipeline-parallel BP on 8 A100 [58]. Mono-Forward's 41% energy saving is MLP-only [60].
- Frozen random layers: up to 22% (IWSLT) and 27% (24-layer WMT) less wall-clock to convergence; RoBERTa LM equal perplexity at equal cost [56].

### 2.6 Cheap recurrent backbones

minGRU is 175x faster per step than a plain-PyTorch GRU at length 512 (T4) but equals Mamba (2.72 vs 2.71 ms); Shakespeare characters: test loss 1.548 vs transformer 1.547 in 575 vs 2,000 steps (steps, not wall-clock; tiny data) [61]. I found no minGRU wall-clock comparison with a FlashAttention transformer under mixed precision. Griffin (recurrence plus local attention, like the lab's backbone): step time equal to an MQA transformer at 2K, transformer slower as length grows, more so at small sizes; 7B/14B match Llama-2 on ~7x fewer tokens, with different data and tuning per the paper [62]. MambaByte beats byte Transformers at matched compute and equals subword Mamba, which reaches near-optimal 2.2x faster [63].

## Part 3 - Origin brief

It renders (conversation embedded in the page JSON; WebFetch alone returned only the title): "Fly Brain Text Generation", 12 Sep 2026 [64]. Proposed: the FlyWire connectome (139,255 neurons; the paper gives 5x10^7 synapses [9]) as a fixed sparse recurrent byte-level LM; frozen topology in V0 with trained edge strengths, 4-state neurons, cell-type-specific updates (Kenyon, dopaminergic, projection), BPTT, e-prop or STDP-plus-dopamine as later options, random-wiring and GRU controls, then structural plasticity, neurogenesis and a terminal environment. Its suggested first experiment, frozen fly plus readout against a control, has since been run publicly: null or tiny effects [22][23], and a small validation edge for real over permuted wiring in an overfitted TinyStories reservoir [24]. The mushroom body appears only in passing (about 2,000 KCs, dopamine-modulated compartments). It did not propose the frozen top-k address or outer-product fast synapses. Its cited URLs: FlyWire, quantization papers, BrainTrace (claim mapping stripped; unverified).

## What this implies for the lab (my inference, not sourced)

Fix kernels first (chunkwise, fused). Byte bigrams (65,536) and trigrams (16.8M) can be direct-indexed, no hashing. Benchmark at equal wall-clock against a speedrun-grade baseline, which got 67x cheaper in about 27 months [47]. Do not expect backprop-free rules to help at 16-50M parameters [53][59]. A frozen address from raw bytes can be precomputed; one from learned hidden states still passes activation gradients.

## Not verified or flagged

Kanerva 1988; Tully-Quinn single-trial figures (not opened, none claimed); the Science version of FlyHash (preprint read); Modi 2020 (abstract only); GLA and DeltaNet figure values; NNCP v3.2 hardware; leakage in [42]; energy for any fly-inspired design; Raven training speed (none in text). [22], [23], [24], [58] and [59] are unreviewed.

## Sources

[1] Caron 2013, https://pmc.ncbi.nlm.nih.gov/articles/PMC4148081/
[2] Aso 2014 (architecture), https://elifesciences.org/articles/04577
[3] Hige 2015, https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/
[4] Litwin-Kumar 2017, https://pmc.ncbi.nlm.nih.gov/articles/PMC5379477/
[5] Dasgupta 2017 (FlyHash), https://www.biorxiv.org/content/10.1101/180471v1 ; Science https://doi.org/10.1126/science.aam9868
[6] Dasgupta 2018 (novelty, fly Bloom filter: ~20 active KCs, 5-10% of 350), https://pmc.ncbi.nlm.nih.gov/articles/PMC6304992/
[7] Modi, Shuai, Turner 2020 (abstract only), https://pubmed.ncbi.nlm.nih.gov/32283995/
[8] Li 2020 (hemibrain MB), https://elifesciences.org/articles/62576
[9] Dorkenwald 2024, https://www.nature.com/articles/s41586-024-07558-y ; Schlegel 2024 (MB homeostasis), https://www.nature.com/articles/s41586-024-07686-5
[10] Zheng 2022, https://pmc.ncbi.nlm.nih.gov/articles/PMC9413950/
[11] Turner 2008, https://pubmed.ncbi.nlm.nih.gov/18094099/
[12] Lin 2014 (APL), https://pmc.ncbi.nlm.nih.gov/articles/PMC4000970/
[13] Aso and Rubin 2016, https://elifesciences.org/articles/16135
[14] Handler 2019, https://pubmed.ncbi.nlm.nih.gov/31230716/
[15] Placais 2017, https://www.nature.com/articles/ncomms15510
[16] Nature 2026, sugar sensor and consolidation, https://www.nature.com/articles/s41586-026-10306-z
[17] Aso 2012 (three dopamine pathways, different stability), https://pmc.ncbi.nlm.nih.gov/articles/PMC3395599/
[18] Liang 2021 (FlyVec), https://arxiv.org/abs/2101.06887
[19] Krotov and Hopfield 2019, https://arxiv.org/abs/1806.10181
[20] Ryali 2020 (BioHash), https://arxiv.org/abs/2001.04907
[21] Shen 2021 (FlyModel), https://arxiv.org/abs/2107.07617
[22] Flies Are All You Need, https://artificialscientific.com/papers/flies-are-all-you-need ; https://github.com/nftechie/flm
[23] FlyGPT, https://github.com/QuixiAI/FlyGPT
[24] fly-llm-hf, https://huggingface.co/ngxson/fly-llm-hf ; gpt-fly, https://huggingface.co/eob/gpt-fly
[25] Connectome reservoir 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/
[26] BrainTrace 2026, https://www.nature.com/articles/s41467-026-68453-w
[27] Bricken and Pehlevan 2021, https://arxiv.org/abs/2111.05498
[28] Bricken 2023, https://arxiv.org/abs/2303.11934
[29] Ramsauer 2020, https://arxiv.org/abs/2008.02217
[30] Ba 2016, https://arxiv.org/abs/1610.06258
[31] Schlag 2021, https://arxiv.org/abs/2102.11174
[32] GLA, https://arxiv.org/abs/2312.06635
[33] DeltaNet, https://arxiv.org/abs/2406.06484
[34] Gated DeltaNet, https://arxiv.org/abs/2412.06464
[35] TTT, https://arxiv.org/abs/2407.04620
[36] Raven, https://arxiv.org/abs/2607.25357
[37] Grave 2017, https://arxiv.org/abs/1612.04426
[38] kNN-LM, https://arxiv.org/abs/1911.00172
[39] Efficient kNN-LM, https://arxiv.org/abs/2109.04212
[40] infini-gram, https://arxiv.org/abs/2401.17377
[41] RETRO, https://arxiv.org/abs/2112.04426
[42] RETRO study, https://arxiv.org/abs/2304.06762
[43] Memorizing Transformers, https://arxiv.org/abs/2203.08913
[44] Hash Layers, https://arxiv.org/abs/2106.04426
[46] Engram, https://arxiv.org/abs/2601.07372
[47] modded-nanogpt, https://github.com/KellerJordan/modded-nanogpt
[48] Large Text Compression Benchmark, https://mattmahoney.net/dc/text.html
[49] cmix, http://www.byronknoll.com/cmix.html
[50] NNCP, https://bellard.org/nncp/nncp_v2.1.pdf ; https://bellard.org/nncp/
[51] Silesia, http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia
[52] Launay 2020 (DFA), https://arxiv.org/abs/2006.12878
[53] Filipovich 2022, https://arxiv.org/abs/2210.14593
[54] Hinton 2022, https://arxiv.org/abs/2212.13345
[55] Bartunov 2018, https://arxiv.org/abs/1807.04587
[56] Reservoir Transformers, https://arxiv.org/abs/2012.15045
[57] EGGROLL, https://arxiv.org/abs/2511.16652
[58] SOLO, https://arxiv.org/abs/2609.35440
[59] Split forward gradients, https://arxiv.org/abs/2607.16612
[60] Mono-Forward evaluation, https://arxiv.org/abs/2511.01061
[61] minGRU, https://arxiv.org/abs/2410.01201
[62] Griffin, https://arxiv.org/abs/2402.19427
[63] MambaByte, https://arxiv.org/abs/2401.13660
[64] Origin brief, a private ChatGPT conversation
