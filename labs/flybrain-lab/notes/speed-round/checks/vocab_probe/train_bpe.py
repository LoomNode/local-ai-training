# Train byte-level BPE tokenizers of several vocabulary sizes on the first 100 MB of the package corpus (the hour arms' train slice).
import sys, time
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
V = int(sys.argv[1]); src = sys.argv[2]; out = sys.argv[3]
def lines():
    with open(src, "rb") as f:
        for ln in f:
            yield ln.decode("utf-8", errors="ignore")
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
tok.decoder = decoders.ByteLevel()
tr = trainers.BpeTrainer(vocab_size=V, min_frequency=2, show_progress=False, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
t0 = time.time(); tok.train_from_iterator(lines(), tr); tok.save(out)
print(f"V={V}: {tok.get_vocab_size()} tokens in {time.time() - t0:.0f} s -> {out}")
