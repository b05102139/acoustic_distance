import librosa
from speechpy.processing import cmvn

def acoustic_distance(file1, file2):
  file1, sr1 = librosa.load(file1)
  file2, sr2 = librosa.load(file2)
  preemph1 = librosa.effects.preemphasis(file1)
  preemph2 = librosa.effects.preemphasis(file2)
  mfcc1 = librosa.feature.mfcc(y=file1, sr=sr1, dct_type=3, n_mfcc=12, hop_length=int(0.010*sr1), n_fft=int(0.025*sr1))
  mfcc2 = librosa.feature.mfcc(y=file2, sr=sr2, dct_type=3, n_mfcc=12, hop_length=int(0.010*sr2), n_fft=int(0.025*sr2))
  mfcc1 = cmvn(mfcc1, variance_normalization=True)
  mfcc2 = cmvn(mfcc2, variance_normalization=True)
  dist, cost = librosa.sequence.dtw(mfcc1, mfcc2)
  return (dist[cost[-1,0], cost[-1,1]]) / (mfcc1.shape[1] + mfcc2.shape[1])
