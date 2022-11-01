from ogb.lsc import WikiKG90Mv2Dataset
import numpy as np
dataset = WikiKG90Mv2Dataset(root = "/data/ogb/ogbl_wikikg2/smore_folder/")
print(dataset.num_entities) 
subset_train_hrt = dataset.train_hrt # numpy ndarray of shape (num_triples, 3)
entity_feat= dataset.entity_feat
import pandas as pd
subset_train_hrt = np.array(subset_train_hrt).astype('int64')
pd.DataFrame(subset_train_hrt[:,[0,2]]).to_csv("/data/ogb/ogbl_wikikg2/smore_folder/raw/edge.csv", header=None, index=None)
pd.DataFrame(subset_train_hrt[:,[1]]).to_csv("/data/ogb/ogbl_wikikg2/smore_folder/raw/edge_reltype.csv", header=None, index=None)
pd.DataFrame(entity_feat).to_csv("/data/ogb/ogbl_wikikg2/smore_folder/raw/node-feat.csv", header=None, index=None)

num_nodes= entity_feat.shape[0]
num_edges= subset_train_hrt.shape[0]

pd.DataFrame(np.array([num_nodes])).to_csv("/data/ogb/ogbl_wikikg2/smore_folder/raw/num-node-list.csv", header=None, index=None)
pd.DataFrame(np.array([num_edges])).to_csv("/data/ogb/ogbl_wikikg2/smore_folder/raw/num-edge-list.csv", header=None, index=None)
i = 1234
# print(train_hrt[i]) # get i-th training triple (h[i], r[i], t[i])
# print(train_hrt.shape)
# valid_task = dataset.valid_dict['h,r->t'] # get a dictionary storing the h,r->t task.
# hr = valid_task['hr']
# t = valid_task['t']
# val_entities=set()
# for hr_item in hr: 
#     val_entities.add(hr_item[0])
# for t_item in t: 
#     val_entities.add(t_item)
# r_pos_dict=dict()

# for hrt_index in range(0,train_hrt.shape[0]):
#     h_ind=train_hrt[hrt_index][0]
#     r_ind=train_hrt[hrt_index][1]
#     t_ind=train_hrt[hrt_index][2]
#     if r_ind in r_pos_dict:
#         r_pos_dict[r_ind].append(h_ind)
#         r_pos_dict[r_ind].append(t_ind)
#     else : 
#         r_pos_dict[r_ind]=[h_ind, t_ind]

# train_ent_set = []
# for r_ind, ent_list in r_pos_dict.items(): 
#     ent_set  = set(ent_list)
#     ent_set_list= list(ent_set)
#     n  = len(ent_set_list)
#     n_sub = n/100
#     train_ent_set.extend(ent_set_list[0:int(n_sub)])

# train_ent_set.extend(list(val_entities))
# train_ent_set= set(train_ent_set)
# import numpy as np
# ent_arr= np.array(list(train_ent_set))
# np.save("/data/ogb/ogbl_wikikg2/smore_folder/sample/ent_ids.npy", ent_arr)
# ent_arr_sorted= np.sort(ent_arr)
# entity_feat = dataset.all_entity_feat
# print("done")
# entity_feat_subset= entity_feat[ent_arr_sorted,:]
# np.save("/data/ogb/ogbl_wikikg2/smore_folder/sample/wikikg90m-v2/processed/entity_feat.npy", np.array(entity_feat_subset))
# print(entity_feat_subset.shape)

# ent_arr_sorted= np.sort(ent_arr)
# ent_id_dict={}
# for i in range(len(ent_arr_sorted)) : 
#     ent_id_dict[ent_arr_sorted[i]]= i

# subset_train_hrt=[]
# train_ent_set = set(train_ent_set)
# for hrt_index in range(0,train_hrt.shape[0]):
#     h_ind=train_hrt[hrt_index][0]
#     r_ind=train_hrt[hrt_index][1]
#     t_ind=train_hrt[hrt_index][2]
#     if (h_ind in train_ent_set)  and ( t_ind in train_ent_set) : 
    
#         subset_train_hrt.append([ent_id_dict[h_ind], r_ind , ent_id_dict[t_ind]])
# np.save("/data/ogb/ogbl_wikikg2/smore_folder/sample/wikikg90m-v2/processed/train_hrt.npy", np.array(subset_train_hrt))

# import pandas as pd
# subset_train_hrt = np.array(subset_train_hrt).astype('int64')
# pd.DataFrame(subset_train_hrt[:,[0,2]]).to_csv("/data/ogb/ogbl_wikikg2/smore_folder/sample/raw/edge.csv", header=None, index=None)
# pd.DataFrame(subset_train_hrt[:,[1]]).to_csv("/data/ogb/ogbl_wikikg2/smore_folder/sample/raw/edge_reltype.csv", header=None, index=None)
# pd.DataFrame(entity_feat_subset).to_csv("/data/ogb/ogbl_wikikg2/smore_folder/sample/raw/node-feat.csv", header=None, index=None)

# num_nodes= len(ent_id_dict)
# num_edges= subset_train_hrt.shape[0]

# pd.DataFrame(np.array([num_nodes])).to_csv("/data/ogb/ogbl_wikikg2/smore_folder/sample/raw/num-node-list.csv", header=None, index=None)
# pd.DataFrame(np.array([num_edges])).to_csv("/data/ogb/ogbl_wikikg2/smore_folder/sample/raw/num-edge-list.csv", header=None, index=None)
# import numpy as np

# ent_ids= np.load("/data/ogb/ogbl_wikikg2/smore_folder/sample/ent_ids.npy")

# train_ent_set= set(ent_ids)
# ent_arr_sorted= np.sort(np.array(ent_ids))
# ent_id_dict={}
# for i in range(len(ent_arr_sorted)) : 
#     ent_id_dict[ent_arr_sorted[i]]= i

# import random
# from ogb.lsc import WikiKG90Mv2Dataset
# dataset = WikiKG90Mv2Dataset(root = "/data/ogb/ogbl_wikikg2/smore_folder/")

# val_t_candidate= np.load("/data/ogb/ogbl_wikikg2/smore_folder//wikikg90m-v2/processed/val_t_candidate.npy")
# val_t= np.load("/data/ogb/ogbl_wikikg2/smore_folder//wikikg90m-v2/processed/val_t.npy")

# valid_task = dataset.valid_dict['h,r->t'] # get a dictionary storing the h,r->t task.
# valid_hr = valid_task['hr']
# t = valid_task['t']

# valid_head=[]
# valid_relation=[]
# valid_tail=[]
# valid_tail_neg=[]

# for hrt_index in range(0,valid_hr.shape[0]):
# h_ind=valid_hr[hrt_index][0]
# r_ind=valid_hr[hrt_index][1]
# t_ind=val_t[hrt_index]
    
# neg_list=[]
# valid_head.append(h_ind)
# valid_tail.append(t_ind)
# valid_relation.append(r_ind)

# # if (h_ind in train_ent_set)  and ( t_ind in train_ent_set) : 
# #     valid_head.append(ent_id_dict[h_ind])
# #     valid_tail.append(ent_id_dict[t_ind])
# #     valid_relation.append(r_ind)
# for cand in val_t_candidate[hrt_index]: 
#     if(cand != t_ind):
#         neg_list.append(cand)
# if(len(neg_list)<500): 
#     randomlist = random.sample(range(0, 1000000),500-len(neg_list))
#     neg_list.extend(randomlist)
# print(len(neg_list))
# valid_tail_neg.append(neg_list[0:500])
# import  torch

# valid_dict= {"head": torch.tensor(valid_head), \
# "relation": torch.tensor(valid_relation), \
#     "tail": torch.tensor(valid_tail), \
#         "tail_neg": torch.tensor(valid_tail_neg), 

# }

# torch.save(valid_dict,"/data/ogb/ogbl_wikikg2/smore_folder//raw/split/time/valid.pt")




    
#         subset_train_hrt.append([ent_id_dict[h_ind], r_ind , ent_id_dict[t_ind]])

# import numpy as np
# import random
# import torch
# subset_train_hrt = np.load("/data/ogb/ogbl_wikikg2/smore_folder//wikikg90m-v2/processed/train_hrt.npy")
# rand_indices= random.sample(range(0, subset_train_hrt.shape[0]),50000)
# train_hr_sample= subset_train_hrt[rand_indices,:]

# train_eval_dict= {"head": torch.tensor(train_hr_sample[:,0]), \
# "relation": torch.tensor(train_hr_sample[:,1]), \
#     "tail": torch.tensor(train_hr_sample[:,2]), \
# }

# torch.save(train_eval_dict,"/data/ogb/ogbl_wikikg2/smore_folder//raw/split/time/train.pt")

# rel_feat = np.load("/data/ogb/ogbl_wikikg2/smore_folder//wikikg90m-v2/processed/relation_feat.npy")
# torch.save(torch.tensor(rel_feat, dtype=torch.float32),"/data/ogb/ogbl_wikikg2/smore_folder//rel_feat.pt")

