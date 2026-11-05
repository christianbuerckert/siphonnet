# Two moons (n=1000, noise=0.2, 50/50 split, 5 seeds)

Test accuracy in % (mean ± std over seeds). N = number of siphons; features = discharge counts in 3 time windows (3N), '+ levels' adds mean fill levels (6N).

| model | N=10 | N=30 | N=100 | N=300 | N=1000 |
|---|---|---|---|---|---|
| abstract SiphonNet | 93.0 ± 0.9 | 95.2 ± 1.0 | 96.4 ± 0.5 | 96.7 ± 0.5 | 96.9 ± 0.6 |
| abstract SiphonNet, W=0 | 94.5 ± 0.8 | 96.2 ± 0.8 | 96.6 ± 0.9 | 96.8 ± 0.8 | 96.9 ± 0.7 |
| physical SiphonNet | 94.8 ± 1.0 | 95.7 ± 1.0 | 96.7 ± 0.6 | 96.8 ± 0.7 | 97.0 ± 0.7 |
| physical SiphonNet, W=0 | 93.6 ± 0.9 | 96.2 ± 0.9 | 96.4 ± 0.8 | 96.6 ± 0.4 | 96.8 ± 0.5 |
| physical SiphonNet + levels | 96.4 ± 0.9 | 96.6 ± 0.8 | 96.9 ± 0.8 | 97.0 ± 0.6 | 96.8 ± 0.5 |
| ELM (random ReLU, 3N features) | 94.0 ± 2.7 | 96.1 ± 0.5 | 96.9 ± 0.7 | 96.8 ± 0.7 | 96.5 ± 0.8 |

| baseline without siphons | accuracy |
|---|---|
| logistic regression (raw x) | 86.7 ± 1.1 |
| polynomial deg 3 + logistic | 94.3 ± 1.2 |
| RBF-SVM | 96.5 ± 1.0 |
