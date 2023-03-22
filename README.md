# SCCN-LoG
Codes for paper "Efficient Representation Learning for Higher-Order Data with Simplicial Complexes" (https://proceedings.mlr.press/v198/yang22a.html)

```
@inproceedings{yang2022efficient,
  title={Efficient Representation Learning for Higher-Order Data with Simplicial Complexes},
  author={Yang, Ruochen and Sala, Frederic and Bogdan, Paul},
  booktitle={Learning on Graphs Conference},
  pages={13--1},
  year={2022},
  organization={PMLR}
}
```

### Step 1: Build simplicial complexes from Cora
Run ```load_cora.ipynb```

### Step 2: Simplex classification on Cora with SCN and SCCN model
```
bash exp.sh 
```
