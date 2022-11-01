data_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full"
model_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/saved_models/TransE_l2_wikikg90m_shallow_d_600_g_10.00"
save_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/saved_models/TransE_l2_wikikg90m_shallow_d_600_g_10.00"

python eval.py --model_name TransE_l2 --data_path $data_path --dataset wikikg90m \
--neg_sample_size_eval 200 --batch_size_eval 200 --model_path $model_path  --save_path $save_path -g 10.0 --LRE --LRE_rank 200
# --encoder_model_name concat