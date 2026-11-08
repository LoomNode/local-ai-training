Vocabulary probe for model #2 (2026-10-15 evening, ledger R110). Four byte-level BPE tokenizers (notes/speed-round/checks/vocab_probe/train_bpe.py,
HF tokenizers, trained on the first 100 MB of corpora/pkgs.bytes, 15 s each) and Qwen3.8-27B's tokenizer (llama-tokenize, vocab-only load of the
GGUF) over the 613 clean package dev windows (results/tables/big/clean_dev.txt at the package dev offset 5,283,527,978; 1,255,424 bytes. Corrected 2026-10-16 09:30: the first run read the windows at 202,117,357, the kernel corpus offset, inside the package train split: 8k 2.77, 16k 2.93, 32k 3.04, 64k 3.11, Qwen 3.32; the call is the same) and the two held-out streams
(corpora/pkgs-tables/{vox28,lab28}/heldout.bytes). vocab_measure.txt: bytes/token and the 12x512 reader's compute per byte at each vocabulary
(measure.py); qwen_counts.txt: Qwen's token counts. Kyle's call: 32k.
