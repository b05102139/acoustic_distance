# MFCC-based Acoustic Distance

This module is a Python implementation of the acoustic distance as described in Bartelds et al. (2020). At its core, it applies dynamic time warping upon audio that is represented as Mel-frequency cepstral coefficients, which additionally go through a number or pre- and post-processing steps. The module can be imported and used as below, where the audio should be wav files:

```python
from acoustic_distance.acoustic_distance import acoustic_distance

acoustic_distance("C:/Users/USER/Downloads/ipa_vowels/a.wav", "C:/Users/USER/Downloads/ipa_vowels/e.wav")
```

Where the result would yield 10.494419525578806.

# References
Bartelds, M., Richter, C., Liberman, M., & Wieling, M. (2020). A new acoustic-based pronunciation distance measure. Frontiers in Artificial Intelligence, 3, 39.
