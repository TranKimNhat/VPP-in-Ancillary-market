Decision letter (Initial Submission)
Safe Multi-Agent Graph Learning for Fast Frequency Response of Virtual Power Plants in Inverter-Based Islanded Microgrids
Subject	Paper TSG-01727-2026, Safe Multi-Agent Graph Learning for Fast Frequency Response of Virtual Power Plants in Inverter-Based Islanded Microgrids
Date sent	22 August 2026 at 14:52 GMT+7
From	
c.koster@ieee.org
To	
fotuhi.ieee@gmail.com, fotuhi@sharif.edu
CC	
fotuhi.ieee@gmail.com, fotuhi@sharif.edu
We regret to advise you that the Reviewing Committee is unable to accept the subject paper for publication as a PES journal paper.  

Please find enclosed the comments of the reviewers which informed the editors when making this recommendation. I hope you will find the explanations satisfactory.

Dr. Mahmud Fotuhi-Firuzabad, Editor in Chief,
IEEE Transactions on Smart Grid

COMMENTS TO THE AUTHORS:
Editor's Comments:

Editor: 1
Comments to the Author:
Although reviewers find the problem under study to be timely and important, they have provided critical feedback regarding several important aspects of the submitted manuscript. These include insufficient justification of novelty and quality of purported contributions, modelling assumptions lacking consistency with inverter-based systems, various questionable technical and performance claims, along with potentially unfair and irreproducible numerical case studies. After synthesizing comments from two reviewers along with careful consideration of the manuscript, I believe that it is not ready to be considered for publication at this time. Authors may find the detailed feedback from reviewers helpful in reshaping and improving their work.

Reviewers' Comments:

Reviewer: 1

Comments to the Author
This manuscript attempts to combine an inductive GraphSAGE encoder with multi‑agent reinforcement learning (MAPPO) to address fast frequency response (FFR) under feeder reconfiguration in islanded microgrids. The engineering context is relevant, but the proposed framework is essentially an assembly of off‑the‑shelf techniques, lacks a clear scientific breakthrough, relies on a linearised dynamic model that is not validated for inverter‑based systems, and uses an oversimplified generalisation test that cannot support the claimed “zero‑shot topology transfer”. Moreover, several baseline comparisons are unfairly configured, which artificially inflates the reported advantages.

1. Both the inductive GraphSAGE encoder and the MAPPO algorithm are mature techniques, and their combination has already appeared in several power‑system applications (e.g., voltage control, fault restoration, economic dispatch) in recent IEEE Transactions papers (2024–2025). The present paper merely transplants this generic framework to the FFR setting and introduces a “joint power‑reference + droop‑gain” action space. However, the necessity of this joint action space is argued only through qualitative comparisons with a fixed droop controller, without a rigorous mechanistic analysis or theoretical justification. Compared with existing literature, the paper does not reveal a new research problem or propose a novel learning paradigm; its contribution remains at the level of yet another engineering combination, which does not constitute the significant methodological advance required by this journal.

2. The state‑space model in (7)–(8) is built on the classical swing equation and Kron reduction, which are appropriate for synchronous‑machine‑dominated systems. However, in 100% inverter‑based microgrids, virtual inertia and damping are control‑programmable parameters, and the resulting dynamics can be highly nonlinear, with fast transients, saturation, and mode switching. The authors provide no comparison against a detailed switched/EMT‑type benchmark to validate the reduced‑order model. Without such validation, the reported RoCoF and nadir values, obtained purely from a linearised model, are not credible. One cannot ascertain whether the reduced model remains sufficiently accurate under large topology changes or severe disturbances, and this undermines the entire simulation study.

3. The common Lyapunov condition in (29) is checked only at the vertices of the gain box, but the manuscript does not report whether a positive‑definite matrix P satisfying all vertex inequalities is actually found. Even if the vertex conditions hold, convexity of the Hurwitz set is not guaranteed, so the box‑wide stability conclusion is not justified. The subsequent average dwell‑time argument relies on the strong assumption that all 96 frozen‑time vertex matrices are Hurwitz, and the computed bound (41.6 s) offers only a modest margin over the typical reconfiguration interval. Furthermore, the one‑step projection in (32) is based on a linearised COI prediction; under large disturbances or topology changes, this projection can be misdirected, potentially causing frequency violations. These theoretical gaps render the “safety envelope” claim weak and insufficient to guarantee real‑world security.

4. All 24 test topologies are generated from the same base feeder through tie‑switch operations and N‑1 contingencies, with the same number of nodes, buses, DERs, and types, and a Jaccard edge distance ≤ 0.18. This amounts to small‑scale perturbations, not a genuine topological transfer. True generalisation would require testing on feeders of different sizes (e.g., migrating from the 123‑bus system to 34‑bus or 85‑bus systems) or with significantly different DER compositions. The paper offers no such evidence, yet it strongly markets “zero‑shot topology generalisation” as its central claim. Even under this mild setting, the proposed method exhibits considerable variance in ITAE (\(\pm 36.6\) Hz·s²), indicating that its robustness is not fully convincing.

5.The GCNN‑PPO baseline is labelled as a “centralised single‑agent” method, which is inherently less scalable than the CTDE architecture used by the proposed method. Its inferior performance is therefore an artefact of the structural design rather than a fair reflection of the graph encoder’s capability.  
- No comparison with stronger graph encoders (e.g., GAT or Graph Transformer) is provided, so the claimed superiority of GraphSAGE is not well supported.  
- For the fixed‑droop baseline, it is unclear whether the gains are re‑tuned for each test topology or kept at the training‑topology values. If the latter, the 3.7× ITAE gap is almost unavoidable and conveys little insight.  
- Moreover, the exceptionally large performance gap between MLP‑MAPPO and the proposed method in several scenarios (e.g., Table VII, S3: 128.7 vs. 57.2 €/MWh) suggests that the reward design or hyperparameter tuning may be unfavourable to the MLP encoder, rather than reflecting a genuine advantage of GraphSAGE.

6. The DSO procurement cost reported in Table VII and Figure 8 depends on energy and capacity price signals λ^"imp"  and λ^"cap" , but neither their reference values nor their actual numerical values are given in the text. The paper mentions three possible VOLL levels (1500, 3000, 8000) but does not present the corresponding results. Consequently, the claimed cost advantage cannot be reproduced or verified under different market conditions. If the ranking is sensitive to these price parameters, the economic conclusion would have limited practical value.

7. Simulation settings are inadequately described, compromising reproducibility  
- The islanded operation scenarios, including load models, line parameters, disturbance timing, and duration, are not fully specified.  
- How are the different contingencies and topologies combined across the 300‑step episodes? How many random seeds are used per test case? These details are missing, making it impossible to assess the statistical significance of the results.  
- The statement “zero frequency violations” is not qualified by whether it covers all 24 topologies × 4 contingency types × multiple seeds, or only selected representative runs.

8. Several arguments are repeated verbatim multiple times (e.g., the “topology‑blind” criticism appears almost identically three times in the introduction). Section III‑A, on the reward function, is overly long with a dozen equations, most of which are standard constructs (dead‑zone, projection, normalisation) that could be moved to an appendix. In contrast, the physical interpretation of the GraphSAGE embeddings and the convergence behaviour of the training are only briefly discussed. This disproportionate allocation of space not only impairs readability but also suggests that the authors have not distilled their core contribution effectively.


Reviewer: 2

Comments to the Author
This paper investigates the provision of fast frequency response (FFR) by a virtual power plant (VPP) under reconfiguration of an all-inverter-based islanded microgrid. Generally, there are several aspects require deeper investigation and rigorous validation. Detailed comments are as follows:
1.The abstract reports the errors as “1.45–1.75 times” and “3.7 times higher,” whereas different ranges of error multiples are used in the main text and conclusion.
2.The Introduction does not clearly explain why GraphSAGE is selected over other message-passing neural networks. Equation (4) only provides an abstract functional representation without specifying a practical computational model.
3.In (14), the power channel is limited to ±10% of the rated power, and a justification for this setting should be provided. In addition, it should be clarified why a 1-s control time step is sufficient for investigating sub-second FFR. Does the 100-ms time step merely represent a finer numerical discretization of the same linear model?
4.How can it be demonstrated that the projected aggregate power in (32) can be physically realized by DERs subject to SoC, rated-power, ramp-rate, and network constraints?
5.How are the 96 vertices constructed?
6.The numerical values used in Equations (17)–(26) are not provided in the subsequent tables.
7.How can reliable 95% confidence intervals and extremely small Wilcoxon p-values be obtained using only three random seeds?
8.Does GraphSAGE-MAPPO still exhibit a significant advantage over the other methods when the common safety projection is removed? How much of the performance improvement can be attributed to the joint power–droop action compared with using either action alone?
9.The manuscript should be carefully proofread for grammatical issues. For example, the Introduction contains many overly long and complex sentences.