# THOR
ICDM 2022: THOR: Self-Supervised Temporal Knowledge Graph Embedding via Three-Tower Graph Convolutional Networks

Way to preprocess dataset
1. cd data
2. python data.py
3. python construct_graph.py

Just for training ICEWS14 right away
1. python train.py

Test model trained with ICEWS14
1. check best epoch via printed output
2. python test.py

Thank you


### Cite
We encourage you to cite our paper if you have used the code in your work. You can use the following BibTex citation:
```
@inproceedings{lee20sigir,
  author   = {Yeon{-}Chang Lee and JaeHyun Lee and DongWon Lee and Sang{-}Wook Kim},
  title     = {THOR: Self-Supervised Temporal Knowledge Graph Embedding via Three-Tower Graph Convolutional Networks},
  booktitle = {},      
  year      = {2022}
}
