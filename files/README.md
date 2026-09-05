---
extra_gated_prompt: Please read Apache License, Version 2.0 before downloading this model.
extra_gated_fields:
  Country: country
  Affiliation: text
  I agree ALL the statements in Apache License, Version 2: checkbox
extra_gated_button_content: Acknowledge license
license: apache-2.0
language:
- ja
pipeline_tag: feature-extraction
tags:
- hubert
- speech
---

# `imprt/kushinada-hubert-large`

This is a Japanese HuBERT Large model pre-trained using 62215 hours of audio extracted from large-scale Japanese TV broadcast audio data by voice activity detection.  
This model was trained using code from the [official repository](https://github.com/facebookresearch/fairseq/).  


## Usage
```python
import soundfile as sf
from transformers import AutoFeatureExtractor
model = "imprt/kushinada-hubert-large"
feature_extractor = AutoFeatureExtractor.from_pretrained(model)
audio_file="/path/to/16k_audio_file"
audio_input, sr = sf.read(audio_file)
feature_extractor(audio_input, sampling_rate=sr)
```

## References
```bibtex
@article{journals/corr/abs-2106-07447,
  added-at = {2021-06-16T00:00:00.000+0200},
  author = {Hsu, Wei-Ning and Bolte, Benjamin and Tsai, Yao-Hung Hubert and Lakhotia, Kushal and Salakhutdinov, Ruslan and Mohamed, Abdelrahman},
  biburl = {https://www.bibsonomy.org/bibtex/2435bd8c9ac37a4eab204ded15e9f8918/dblp},
  ee = {https://arxiv.org/abs/2106.07447},
  interhash = {c85407653eddc9c9256c261afe8d6954},
  intrahash = {435bd8c9ac37a4eab204ded15e9f8918},
  journal = {CoRR},
  keywords = {dblp},
  timestamp = {2024-04-08T22:55:35.000+0200},
  title = {HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.},
  url = {http://dblp.uni-trier.de/db/journals/corr/corr2106.html#abs-2106-07447},
  volume = {abs/2106.07447},
  year = 2021
}
```

## License

[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
