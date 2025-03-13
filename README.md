# DT-GAT
This is the code repository for the paper "Dual-Channel Time-Aware Graph Attention Network for Session-Based Recommendation"





## Abstract
Session-based recommender systems face significant challenges in accurately predicting user preferences due to the limited availability of long-term historical interactions. While recent advances in deep learning and graph-based approaches have improved recommendation performance, the temporal aspects of user interactions remain underutilized. This paper identifies three critical temporal challenges in session-based recommendations: interest shifts indicated by long intervals between interactions, interaction noise from brief engagements, and system popularity effects during high-traffic periods. To address these challenges, we propose a novel Dual-channel Time-aware Graph Attention Network (DT-GAT) to incorporate temporal information into session representations from both user and session perspectives. Specifically, the user-wise learning channel employs a temporal graph attention network to capture interest shifts and filter interaction noise, while the session-wise learning channel utilizes a temporal graph attention network to handle inconsistent popularity trends. Additionally, we introduce a multi-temporal window processing mechanism to construct robust session representations that effectively capture short-term interests while filtering noise. Extensive experiments on three real-world datasets demonstrate that DT-GAT significantly outperforms state-of-the-art baseline models.

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
xxxx
```
