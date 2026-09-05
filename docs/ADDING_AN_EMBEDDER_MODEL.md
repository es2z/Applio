# Adding an Embedder Model

Written after adding `japanese-hubert-large` (1024-dim / 24 layers), the first embedder
here that is not a 768-dim HuBERT Base. Read this before adding another one.

Everything below was measured on this machine, not inferred. Where a number appears, the
recipe that produced it is in [Measuring a new embedder](#measuring-a-new-embedder).

---

## 1. The short version

Adding an embedder is one dict entry plus the choice lists. Everything that used to be
hardcoded per-embedder is now derived:

```python
# rvc/lib/utils.py:44
EMBEDDERS = {
    "your-embedder": {
        "dir": "your_embedder",          # folder under rvc/models/embedders/
        "repo": "org/your-embedder",     # omit for a legacy IAHispano/Applio .bin embedder
        "revision": "<40-char commit sha>",
        "feature_scale": 1.0,            # omit unless measurement says otherwise (§4)
    },
}
```

Then add the name to the choice lists (**9 copies**, all asserted by
`tests/test_embedder_models.py`):

- `core.py` — `infer`, `batch_infer`, `tts`, `extract` parsers
- `tabs/train/train.py`, `tabs/inference/inference.py` (×2), `tabs/tts/tts.py`,
  `tabs/realtime/realtime.py`
- optional: `assets/Applio_NoUI.ipynb`, `CLAUDE.md`

**Do not** add dimension branches anywhere. The width flows on its own:

| where | how the width is decided |
|---|---|
| extraction | whatever the model outputs; `measure_feature_dim` reads it back off the `.npy` |
| training | `generate_config` writes `text_enc_hidden_dim` into `logs/<model>/config.json` |
| FAISS index | `faiss.index_factory(big_npy.shape[1], ...)` |
| inference / realtime | `checkpoint_text_enc_hidden_dim` reads `enc_p.emb_phone.weight.shape[1]` |

`enc_p.emb_phone` (`rvc/lib/algorithm/encoders.py`) is **the only tensor in the whole
model whose shape depends on the embedder**. That single fact is what makes all of this
cheap, and it is why warm starting from a differently sized pretrain works.

---

## 2. Run this before writing any code

Two hours of debugging is one measurement. Load the candidate next to `contentvec` and
`japanese-hubert-base` and look at four things (§8 has the script):

1. **Frame count.** Must match the existing embedders on the same audio, or every
   downstream length assumption breaks. All HuBERT variants here share the conv frontend
   (`conv_kernel [10,3,3,3,3,2,2]`, `conv_stride [5,2,2,2,2,2,2]`, 320× → 50 fps), so they
   agree at 1742 frames on `logs/reference/reference.wav`. A model with a different
   frontend is a much bigger job than this document covers.
2. **`feat_extract_norm` / `conv_bias` / `do_stable_layer_norm`** from its `config.json`.
   These decide whether §3 applies.
3. **Per-frame norm of `last_hidden_state`** against contentvec's 9.82. This decides
   whether §4 applies.
4. **`|pos|/|h|`** — how much of the encoder input is positional convolution. Not a knob,
   but it is the single number that explained k2 (§5).

---

## 3. Landmine: `do_normalize` is load-bearing for some embedders and not others

Applio has always fed the embedder a **raw waveform**. That was harmless for years because
every embedder here was `feat_extract_norm: "group"` with `conv_bias: false` — the
GroupNorm right after the first bias-free conv cancels any scalar gain on the input.

`japanese-hubert-large` is `feat_extract_norm: "layer"` with `conv_bias: true`. Nothing
cancels the gain, and its `preprocessor_config.json` sets `do_normalize: true`.

Measured effect of skipping the normalisation:

| embedder | feat_extract_norm | conv_bias | feature change if skipped |
|---|---|---|---|
| contentvec | group | false | 0.29% |
| japanese-hubert-base | group | false | 4.99% |
| japanese-hubert-base-k2 | group | false | 0.21% |
| **japanese-hubert-large** | **layer** | **true** | **59.25%** |

`load_embedding` reads `preprocessor_config.json` and records `input_do_normalize` on the
model; `apply_embedder_input_normalization` (`rvc/lib/utils.py:204`) applies it. This is
automatic for any repo that ships that file — **you do not have to do anything**, but you
do have to know it is happening, because it changes the features a model is trained on.

### 3a. The floor, and why it is not the official formula

Zero-mean/unit-variance normalisation over a window has a nasty property in a voice
changer: **it lifts near-silence to speech level.** Measured, -60 dBFS room tone comes out
at 0.95 RMS — a gain of about 60 dB — and the embedder then reads amplified noise as
speech while the decoder synthesises voiced rubbish over the silence.

Training sees peak-normalised 3-second slices; offline inference sees silence-split
segments; realtime sees a rolling convert buffer. Same audio, three different windows.

`EMBEDDER_INPUT_STD_FLOOR` (`rvc/lib/utils.py:91`, default `0.01` ≈ -40 dBFS RMS) clamps
the divisor. Speech is far above it and untouched; a quiet window stays quiet. Measured, it
changes a room-tone window's features by 52% and moves them from cos 0.68 to cos 0.80
against digital silence.

Set it to `0.0` for the literal `Wav2Vec2FeatureExtractor` behaviour. **If you change it,
every existing `do_normalize` model is stale** — which is why it is part of the identity
(§6).

---

## 4. Landmine: feature magnitude vs `emb_pitch`

`TextEncoder.forward` adds the two terms with no scaling between them:

```python
x = self.emb_phone(phone)          # your embedder's features
x += self.emb_pitch(pitch)         # nn.Embedding, default init, no scale freedom
```

RVC v2 was tuned around contentvec's magnitude. An embedder an order of magnitude off
leaves the content term drowned out, and `emb_phone` never catches up because its own
gradient scales with the input magnitude too.

Measured per-frame L2 norm of `last_hidden_state`:

| embedder | norm/frame | scale to match contentvec | shipped `feature_scale` |
|---|---|---|---|
| contentvec | 9.82 | 1.00 | — |
| japanese-hubert-base | 6.35 | 1.55 | — |
| **japanese-hubert-base-k2** | **0.58** | **16.9** | **10.0** |
| japanese-hubert-large | 5.95 | 1.65 | — |

Rule of thumb: **within ~2× of contentvec needs no scale.** k2 at 17× did.

Do not guess this from the architecture. Before measuring, `japanese-hubert-large` was
predicted to land near √1024 ≈ 32 because its `last_hidden_state` is LayerNorm output. It
measures 5.95. The LayerNorm has a learned gain; the dimension tells you nothing.

---

## 5. Two claims about the legacy embedders that are FALSE

Both were written down confidently in this repo and both cost time. They are corrected in
`CLAUDE.md`; repeated here so the next reader does not re-derive them.

### "The legacy embedders silently lose their positional-conv weights"

`from_pretrained(..., output_loading_info=True)` reports, for `contentvec`,
`japanese-hubert-base` **and** `japanese-hubert-large`:

```
missing_keys    : ['encoder.pos_conv_embed.conv.parametrizations.weight.original0', ...]
unexpected_keys : ['encoder.pos_conv_embed.conv.weight_g', 'encoder.pos_conv_embed.conv.weight_v']
```

This reads like the layer is being dropped and re-initialised. **It is not.** transformers
4.44.2 renames those keys during loading; the lists are bookkeeping left over from the
rename. Verified by comparing the loaded tensors against the raw checkpoint:

```
contentvec  weight_g EQUAL=True   weight_v EQUAL=True
japanese-hubert-base  weight_g EQUAL=True   weight_v EQUAL=True
```

**Do not add a remapping shim.** If you are ever unsure, the check is four lines (§8).

### "k2's features look different because of that"

k2's oddity is intrinsic to the checkpoint. Its positional conv is genuinely dominant:

| embedder | `\|pos\|/\|h\|` | pos_conv weight norm |
|---|---|---|
| contentvec | 0.94 | 16.58 |
| japanese-hubert-base | 0.80 | 16.83 |
| **japanese-hubert-base-k2** | **7.30** | **33.85** |
| japanese-hubert-large | 0.90 | 8.89 |

k2 carries tiny hidden states *and* a positional conv ~8× more dominant than anything
else. `feature_scale = 10.0` fixes the magnitude against `emb_pitch`; it cannot change the
ratio of positional to content information inside the features. **If a k2 model sounds
noisy, suspect that ratio, not a missing parameter.**

---

## 6. Landmine: stale artifacts, and the guard that now catches them

Changing the embedder — or its scale, output layer, or input std floor — invalidates
**every `.npy`, the FAISS index, and `enc_p.emb_phone` together**.

Before this was guarded, `extract.py` re-extracted the features and deleted the index, but
left `G_*.pth` / `D_*.pth` in place, and `train.py` resumed from them unconditionally
(inside a bare `except Exception`). The generator's `enc_p.emb_phone` — and the Adam
moments behind it — stayed fitted to the *old* features. That does not raise. It produces
a model that sounds broken and never recovers, and it is the most likely explanation for
the noisy k2 run.

The identity is now stamped everywhere and compared on resume:

| field | why it invalidates features |
|---|---|
| `embedder_model` | different model |
| `embedder_feature_scale` | multiplies the stored features |
| `embedder_output_layer` | different layer |
| `embedder_dim` | different width |
| `embedder_input_std_floor` | changes quiet windows by >50% |

Written to `logs/<model>/model_info.json`, into `G_*.pth`/`D_*.pth` via `save_checkpoint`,
and into the exported `.pth` via `extract_model`. Compared by
`describe_embedder_mismatch` (`rvc/lib/utils.py:315`), enforced by `assert_resumable`
(`rvc/train/utils.py:144`).

**If you add a knob that changes the stored features, add it to that identity.** A record
that names no embedder at all predates tracking and never counts as a mismatch, so old
folders keep working.

---

## 7. Things that need no work, and why

- **The mute file.** `extract.py:249` writes this run's own `logs/<model>/mute.npy` with
  the same embedder, so the silent padding rows always match the batch width. It lives
  outside `extracted/` because `extract_index.py` indexes everything in there and silence
  does not belong in a retrieval index. The shipped `logs/mute*` folders are only a
  fallback for folders extracted before this existed.
- **Warm starting from a 768 pretrain.** `load_pretrained` (`rvc/train/utils.py:95`) skips
  exactly `enc_p.emb_phone.*` on a shape mismatch and inherits encoder, flow, decoder and
  speaker embedding; the discriminator never sees the features and loads whole. Any *other*
  shape mismatch is a real mistake (wrong sample rate or vocoder) and still exits.
  Confirmed working: a 1024-dim run reached a usable model in 220 epochs from `f0G48k.pth`.
- **A deep model's intermediate layers.** For a `do_stable_layer_norm` model, only
  `last_hidden_state` has the final LayerNorm; `hidden_states[i]` are raw pre-norm
  residuals. Measured on japanese-hubert-large: 69 per frame at layer 0 rising to 538 at
  layer 23, against 5.9 for the last layer. `embedder_forward` applies
  `encoder.layer_norm` when a non-final layer is selected. `hidden_states[-1]` **is**
  `last_hidden_state`, so the default path is unaffected.
- **`.eval()`.** `layerdrop: 0.1` and `apply_spec_augment` are gated on `module.training`.
  On a 24-layer model a leak would silently drop whole layers. `_finalize_embedder` calls
  `.eval()` defensively.

---

## 8. Measuring a new embedder

Run from the repo root with `env\python.exe`. Prints everything §2 asks for.

```python
import os, torch, librosa
from torch import nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor

class M(HubertModel):
    def __init__(self, c):
        super().__init__(c)
        self.final_proj = nn.Linear(c.hidden_size, c.classifier_proj_size)

ROOT = os.path.join("rvc", "models", "embedders")
CANDIDATES = {
    "contentvec": dict(pretrained_model_name_or_path=os.path.join(ROOT, "contentvec")),
    "japanese-hubert-base": dict(pretrained_model_name_or_path=os.path.join(ROOT, "japanese_hubert_base")),
    "NEW": dict(pretrained_model_name_or_path="org/your-embedder",
                cache_dir=os.path.join(ROOT, "your_embedder"),
                revision="<sha>", use_safetensors=True),
}

audio, _ = librosa.load("logs/reference/reference.wav", sr=16000)
wav = torch.from_numpy(audio).float().view(1, -1)
norm = lambda x: (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-7)

for name, kw in CANDIDATES.items():
    m, info = M.from_pretrained(**kw, output_loading_info=True); m.eval()
    c = m.config
    with torch.no_grad():
        raw = m(wav)["last_hidden_state"]
        nrm = m(norm(wav))["last_hidden_state"]
        h = m(norm(wav), output_hidden_states=True)["hidden_states"][0]
        pos = m.encoder.pos_conv_embed(h)
    print(f"{name}: dim={c.hidden_size} layers={c.num_hidden_layers} "
          f"norm={c.feat_extract_norm} conv_bias={c.conv_bias} stable_ln={c.do_stable_layer_norm}")
    print(f"   frames={raw.shape[1]} norm/frame={raw[0].norm(dim=-1).mean():.3f} "
          f"|pos|/|h|={pos.norm()/h.norm():.3f}")
    print(f"   do_normalize changes features by "
          f"{(raw - nrm).norm() / raw.norm() * 100:.2f}%")
    # The weight_g/weight_v warning is cosmetic - prove it rather than believing it:
    print(f"   missing={[k for k in info['missing_keys'] if 'final_proj' not in k]}")
```

To prove the pos_conv weights really did load, for a local `.bin` embedder:

```python
raw = torch.load(f"{path}/pytorch_model.bin", map_location="cpu", weights_only=True)
sd = M.from_pretrained(path).state_dict()
torch.equal(raw["encoder.pos_conv_embed.conv.weight_g"].reshape(
    sd["encoder.pos_conv_embed.conv.parametrizations.weight.original0"].shape),
    sd["encoder.pos_conv_embed.conv.parametrizations.weight.original0"])   # True
```

---

## 9. Verification checklist

Run all of it. The unit tests are fast; the end-to-end catches what mocks cannot.

```bat
env\python.exe -m unittest discover -s tests -v
```

1. **Legacy regression.** `embedder_forward(model, x)` must be *bit-identical* to the old
   `model(x)["last_hidden_state"]` for every `input_do_normalize=False` embedder.
   Do not compare rendered audio: **RVC inference samples noise and is not
   deterministic** — two runs of identical code differ by 0.77 relative L2. Compare
   features on CPU instead.
2. **End-to-end at the new width.** preprocess → extract → train 1-2 epochs → infer.
   Check `extracted/*.npy` and `mute.npy` widths, `config.json`'s `text_enc_hidden_dim`,
   the index `d`, `enc_p.emb_phone.weight.shape`, and the metadata in the exported `.pth`.
3. **The resume guard.** Re-extract the same folder with a different embedder while
   `G_*.pth` is still there, then train — it must refuse, not resume.
4. **Realtime through `AudioCallbacks`, not `create_pipeline`.** See §10.
5. **Silence.** Feed audio with leading/trailing room tone and confirm nothing voiced is
   synthesised over it.

---

## 10. The realtime constructor chain will bite you

`AudioCallbacks` → `VoiceChanger` → `Realtime` → `create_pipeline` forward a long list of
settings by hand. When those calls were positional, inserting one parameter into a
signature shifted every argument after it: `vad_frame_ms` received `sid`'s `0` and every
run died with

```
ValueError: VAD frame duration must be 10, 20, or 30 ms
```

— with no relation to the embedder, on every model and setting combination.

They are keyword-only by convention now, and
`tests/test_embedder_models.py::RealtimeWiringTest` fails on any positional call in
`rvc/realtime/`. **Keep it that way.**

The lesson that generalises: `create_pipeline` smoke-tested fine because it was called
directly. Test the entry point the UI actually uses.

---

## 11. Reference numbers (RTX 4090, `logs/reference/reference.wav`, 1742 frames)

| embedder | dim | layers | norm/frame | `\|pos\|/\|h\|` | do_normalize effect | ms / 1.5 s |
|---|---|---|---|---|---|---|
| contentvec | 768 | 12 | 9.82 | 0.94 | 0.29% | — |
| japanese-hubert-base | 768 | 12 | 6.35 | 0.80 | 4.99% | 5.7 |
| japanese-hubert-base-k2 | 768 | 12 | 0.58 | 7.30 | 0.21% | — |
| japanese-hubert-large | 1024 | 24 | 5.95 | 0.90 | 59.25% | 10.4 |

F0 extractors on the same window, for scale: `fcpe` 3.2 ms, `rmvpe` 18.8 ms,
`crepe-full` 25.2 ms, `mangio-crepe-full` 40.6 ms.

**The embedder is not the bottleneck.** Going 768 → 1024 costs under 5 ms; a heavy F0
method costs four times the whole Large embedder. `bf16` measured *slightly slower* than
fp32 (11.2 vs 10.4 ms) because at this size the embedder is kernel-launch bound, not
compute bound — `embedder_precision` exists for slower cards, and bf16 is preferred over
fp16 there since deep pre-norm transformers can overflow in fp16.

---

## 12. Where to look when something is wrong

| symptom | look at |
|---|---|
| shape error in `data_utils.py` collate | mute width vs feature width — `logs/<model>/mute.npy` |
| `size mismatch for enc_p.emb_phone.weight` | expected; `checkpoint_text_enc_hidden_dim` should have prevented it |
| FAISS assert / in-place shape error on retrieval | index built at a different width; the guard should have skipped it |
| output is voiced rubbish over silence | `EMBEDDER_INPUT_STD_FLOOR`, §3a |
| voice cuts out mid-speech, never improves with training | feature magnitude vs `emb_pitch`, §4 |
| trained model sounds broken from the start and never recovers | stale resume, §6 — check the identity in `G_*.pth` |
| realtime dies before touching a model | constructor chain, §10 |
| "some weights were not used / newly initialized" | almost certainly cosmetic, §5 — prove it before acting |

Useful one-liners:

```python
# what a checkpoint was actually trained on
import torch; c = torch.load(p, map_location="cpu", weights_only=True)
{k: c.get(k) for k in ("embedder_model","embedder_feature_scale","embedder_output_layer",
                       "embedder_input_std_floor","text_enc_hidden_dim","sr","vocoder")}
c["weight"]["enc_p.emb_phone.weight"].shape      # the real width, always

# what a run folder was extracted with
import json; json.load(open("logs/<model>/model_info.json"))
import numpy as np; np.load("logs/<model>/mute.npy", mmap_mode="r").shape
import faiss; faiss.read_index("logs/<model>/<model>.index").d
```

---

## 13. Training hyperparameters are separate from the embedder

`learning_rate` and `c_mel` live in `logs/<model>/config.json`, seeded once from
`rvc/configs/<sr>.json` by `generate_config` (`rvc/train/extract/preparing_files.py:11`).
The stock values are unchanged from upstream: lr `1e-4`, `c_mel` 45, `c_kl` 1.

They are exposed in the Training tab and as `--learning_rate` / `--c_mel` on
`core.py train`. The tab reads the selected run's real values via `read_train_settings`
and writes them back with `apply_train_settings`, which only touches keys that were
actually passed **and** actually differ — so handing back what was read is a byte-level
no-op and cannot clobber a hand edit. `c_kl`, `lr_decay`, `segment_size` and everything
else still need editing the JSON directly.

`generate_config` deliberately rewrites **only** `text_enc_hidden_dim` on a re-extract, so
any hand edit to the training block survives.

Which value is right is about the run, not the embedder: `1e-4` is what the pretrained
models were trained at and is what a run starting from one wants — including a warm start
where `enc_p.emb_phone` is freshly random, since a lower rate starves the one layer that
has to learn from scratch. The much lower rates (2e-05, 5e-06) are for continuing a model
that is already trained.
