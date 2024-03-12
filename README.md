# sl-pitchfork
***emulating individual modes of solar-like oscillators using a single branching neural network***\
\
We'll build this repo up as we go (_itallics_ for WIP), but for now, the process is:
- Load in trained emulator including .json dict of useful information (remember to add standardisation and un-standardisation before and after predictions)
- Define priors for fundamental parameters (_uniform bounded to edge of grid_)
- Choose simulated star for posterior recovery (_taken from grid_)
- Run inference using emulator in dynesty to sample posteriors
