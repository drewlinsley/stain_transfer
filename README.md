# Models to classify cell phenotypes during live imaging experiments
Built using Python 3.7.12, Pytorch, and Pytorchlightning.\
Visualization with [Weights&Biases](https://wandb.ai/) 

---

To auto-enable wandb, do the following:
`cp netrc ~/.netrc`

## Supervised classification

*Train models to learn a mapping from cell image to a label (e.g., TDP-43 or control).*

`python run.py --config-name=u19_pilot.yaml`
