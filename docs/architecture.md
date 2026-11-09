# Architecture and included boundary

The research design connects:

1. Image preparation and anatomy-oriented evidence extraction.
2. Candidate 3D planes with AP, cutting axis, tilt and in-plane rotation.
3. Candidate matching with visibility and ambiguity checks.
4. Joint slice-sequence selection and cutting-pose refinement.
5. Abstention and sealed predictions before held-out truth scoring.

Only plane sampling and the numerical sequence decoder are shipped here.
Learned extractors, checkpoint loading, candidate banks, data acquisition,
scheduler operations, truth scoring, and the full abstention mechanism are
not included.

## Plane conventions

Volume axes are (AP, S/I, R/L). PyTorch's 5D sampling grid is ordered
(R/L, S/I, AP). A candidate combines an AP index, in-plane rotation,
cutting-axis angle and tilt. Left, right and whole-volume lateral extents
are explicit arguments.

The sampler accepts a NumPy volume or a PyTorch tensor. The example passes
device="cpu" explicitly. The numerical sampler retains the research
implementation, including its original device-selection behavior.

## Sequence conventions

sequence_decode receives query metadata, a lower-is-better cost matrix,
candidate metadata and an explicit decoder configuration. It selects the
lowest-cost path among each section's supplied top candidate states.
Transitions penalize AP reversals, excessively large forward drops, normal
changes and rotation changes. Standard sections are ordered by source number
within a dataset; alternates are selected per image.

One AP index represents ten micrometres in this convention. The included
synthetic costs and decoder weights are illustrative and are not adopted
scientific thresholds. The decoder reads no truth file.
