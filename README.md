# DT-GAT
This is the code repository for the paper "Dual-Channel Time-Aware Graph Attention Network for Session-Based Recommendation"





## Introduce
Session-based recommendation aims to infer a user's next action from a short and anonymous interaction sequence, where long-term preference histories are often unavailable. Existing neural, graph-based, and hypergraph-based session recommenders have substantially improved the modeling of item transitions and high-order collaborative relations. However, many of them still treat temporal information as an auxiliary feature, rather than as a structured signal that reflects heterogeneous behavioral and system-level dynamics. In practice, temporal patterns in sessions are not uniform. A long interval between two interactions may indicate an interest shift, an extremely short engagement may correspond to noisy or weak feedback, and interactions occurring during high-traffic periods may be affected by transient popularity rather than stable user preference. These observations suggest that temporal information should not be modeled by a single decay function or a global time embedding.

To address this issue, we propose a Dual-channel Time-aware Graph Attention Network (DT-GAT) for session-based recommendation. The central idea is to decompose temporal session modeling into two complementary channels. The user-wise temporal channel captures preference transitions and suppresses noisy interactions by learning attention over time-aware item relations within user behavior sequences. The session-wise temporal channel using a temporal hypergraph to model session-level temporal consistency and popularity fluctuation, allowing the model to distinguish stable preference evidence from traffic-induced effects. In addition, we design a multi-temporal-window processing mechanism to aggregate behavioral evidence under different temporal granularities, thereby improving the robustness of session representation learning. Extensive experiments on three real-world datasets show that DT-GAT consistently outperforms representative sequential, graph-based, and time-aware recommendation baselines.

## Datasets
Datasets is avaiable at: [https://drive.google.com/drive/folders/1zM7Tm5RMH_gX33NA_BHdwtkiQ1Dl9PyR?usp=drive_link](https://drive.google.com/drive/folders/1zM7Tm5RMH_gX33NA_BHdwtkiQ1Dl9PyR?usp=drive_link)

## Run the code: 
```bash
pip install -r requirements.txt  
python main.py --dataset diginetica --intent_num 4
python main.py --dataset Retailrocket --intent_num 3
python main.py --dataset Yoochoose64 --intent_num 4
```

## Citation

```bash
@article{guo2026dual,
  title={Dual-channel time-aware graph attention network for session-based recommendation},
  author={Guo, Linjiang and Wu, Shiqing and Lu, Dan and Gao, Longxiang and Xu, Guandong},
  journal={Information Sciences},
  pages={123289},
  year={2026},
  publisher={Elsevier}
}

```
