transe_path = "/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/saved_models/TransE_l2_wikikg90m_shallow_d_600_g_10.00"
complex_path= "/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/saved_models/ComplEx_wikikg90m_shallow_d_600_g_10.00"
valid_candidates = np.load("/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/wikikg90m-v2/processed/val_t_candidate.npy")
transe_result_dict = torch.load(os.path.join(
                transe_path, "valid_{}_{}.pkl".format(0, "final")), map_location="cpu")
complex_result_dict = torch.load(os.path.join(
                complex_path, "valid_{}_{}.pkl".format(0, "final")), map_location="cpu")
print(result_dict.keys())
key= 't_pred_top10'
index =(result_dict['h,r->t']['t_pred_top10'])
temp = []
for ii in range(index.shape[0]):
    temp.append(valid_candidates[ii][index[ii]])
result_dict['h,r->t'][key] = np.concatenate(np.expand_dims(temp, 0))
print(result_dict['h,r->t']['t_pred_top10'][:,0:10])


import pandas as pd
transe_result_dict['h,r->t']['scores']= transe_result_dict['h,r->t']['scores'].view(15000,-1)
transe_result_dict['h,r->t']['candidates']= transe_result_dict['h,r->t']['candidates'].view(15000,-1)
transe_scores= transe_result_dict['h,r->t']['scores'].numpy()
transe_cands= transe_result_dict['h,r->t']['candidates'].numpy()
ind = np.argsort(transe_cands, axis=1) 
ind= ind.astype(np.int64)
transe_cands_sorted = np.take_along_axis(transe_cands, ind, axis=1)
transe_scores_unsorted= np.take_along_axis(transe_scores, ind, axis=1)
print(transe_cands_sorted[:,0:10])

complex_result_dict['h,r->t']['scores']= complex_result_dict['h,r->t']['scores'].view(15000,-1)
complex_result_dict['h,r->t']['candidates']= complex_result_dict['h,r->t']['candidates'].view(15000,-1)
complex_scores= complex_result_dict['h,r->t']['scores'].numpy()
complex_cands= complex_result_dict['h,r->t']['candidates'].numpy()
ind2 = np.argsort(complex_cands, axis=1) 
# #ind= ind.astype(np.int64)
complex_cands_sorted = np.take_along_axis(complex_cands, ind2, axis=1)
complex_scores_unsorted= np.take_along_axis(complex_scores, ind2, axis=1)
print(complex_scores[0,0:100])
print(complex_scores_unsorted[0,0:100])

from ogb.lsc import WikiKG90Mv2Dataset, WikiKG90Mv2Evaluator
evaluator = WikiKG90Mv2Evaluator()

final_scores = 0.3*complex_scores_unsorted + 0.7* transe_scores_unsorted
print(final_scores.shape)
final_ind= np.argsort(-1*final_scores, axis=1) 
final_cand=  (np.take_along_axis(complex_cands_sorted, final_ind, axis=1))
final_scores_sorted=  (np.take_along_axis(final_scores, final_ind, axis=1))
final_cand_top10= final_cand[:,0:10]
print(final_scores_sorted)

print(final_cand_top10)

# final_scores = 0.7*transe_socres + 0*complex_socres
# print(final_scores.shape)
# print(transe_result_dict['h,r->t']['t'].numpy().shape)
# print(final_cand.shape)
result_dict={}
result_dict['h,r->t']={'t_pred_top10':complex_cands[:,0:10] , 't':transe_result_dict['h,r->t']['t'].numpy()}
metrics = evaluator.eval(result_dict)
metric = 'mrr'
print("valid {}".format(metrics[metric]))

result_dict={}
result_dict['h,r->t']={'t_pred_top10':final_cand_top10 , 't':transe_result_dict['h,r->t']['t'].numpy()}
metrics = evaluator.eval(result_dict)
metric = 'mrr'
print("valid {}".format(metrics[metric]))

# print(complex_cands_sorted[:,0:10])


# print(result_dict['h,r->t']['candidates'][:,0:10])
# arr= np.array(result_dict['h,r->t']['candidates'][:,0:10])
# arr= arr.astype('str')
# np.savetxt('./val_cands.csv', arr, delimiter=',')