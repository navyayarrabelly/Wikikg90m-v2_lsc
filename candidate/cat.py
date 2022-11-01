import numpy as np
import sys

input_prefix= "/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/wikikg90m-v2/processed/test_challenge_outputs/test_challenge_candidate_60000"
output= "/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/wikikg90m-v2/processed/test_challenge_t_candidate_60000.npy"
e2r=[]
for i in range(15):
  e2r.append(np.load(input_prefix + '_%d.npy' % i)[:, 3:])
  print(e2r[-1].shape)
e2r = np.concatenate(e2r, axis=0)
print(e2r.shape)

np.save(output, e2r)
