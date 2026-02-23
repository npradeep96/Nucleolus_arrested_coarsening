## Notes regarding the analysis of the experimental data

Figure 1C, Figure 3C,E, and Figure 4B were generated with the Notebook [Drop_size_analysis.ipynb](Drop_size_analysis.ipynb). 
For the drug inhibition experiments and the starved/fed experiments, in the distance analysis we filtered out distances exceeding 30.0 pixels; the rationale was that these FCs are likely artifacts, since they would correspond to nucleoli containing only a single FC. The filtering did not visibly affect the distribution of distances. 
For the TCOF overexpression experiments, we did not filter out these FCs, because that would significantly change the distribution of distances. 
In the Notebook [2D_Statistics_and_Distribution.ipynb](2D_Statistics_and_Distribution.ipynb) we used a different filtering approach (by filtering out FCs not part of a cluster) on all of these experiments (drug inhibition experiments, starved/fed experiments, and TCOF overexpression experiments) to confirm that the results do not qualitatively depend on the choice of filtering strategy.

Figure 1E was generated with the Notebook [distribution_shape.ipynb](distribution_shape.ipynb).

Figure S1C,D, Figure S2A,B, and Figure S6B,D were generated with the Notebook [3D_Statistics_and_Distribution.ipynb](3D_Statistics_and_Distribution.ipynb), without filtering out any FCs, because grouping by nucleoli was done by hand in these experiments.

Figure S3 was generated with the Notebook [2D_Statistics_and_Distribution.ipynb](2D_Statistics_and_Distribution.ipynb).