# Production-Planning

## Configuration

We use `pyscipopt` for mip modeling and solving, with `scipy` and `tqdm` for instance generation. To configure:
```
conda install --channel conda-forge pyscipopt
pip install scipy tqdm
```

## Benchmark Testing

The two datasets are included in **PP-Global** and **PP-Local** respectively in mps form, a widely-employed form for MIPs. We provide a simple test program in `test.py` to solve with `pyscipopt`. An examplary test is:
```
python test.py --dataset PP-Global --data-idx 1
```
which solves the first instance in PP-Global with scip solver.

## Dataset Generation

We also provide the dataset generation program in `data_gen.py`, facilitating generating data in different distributions and scales. An example is:
```
python data_gen.py -p 800 -o 2400
```
which generates production planning data with around 800 production lines and 2400 order groups in `data/p800_o2400`.