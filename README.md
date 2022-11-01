# Wikikg90m-v2_lsc

1. We generate 60K tail candidates to evaluate for each (head,relation) pair, following the PIE. 
2. We further augment the candidates with CHAI based candidates. 
3. We train 3 seperate models to train the tail node prediction problem for a given KG. 
   Train a TransE model with  shallow embeddings wikikg90m-v2/run_TransE.sh
   Train a CompleX model with  shallow embeddings wikikg90m-v2/run_CompleX.sh
  Train a GNN based architecture with GAT as encoder and TransE as decoder 
4. Ensemble the predictions from above models. 
9.05* GNN_AE + 0.85* TransE + 0.15* CompleX
