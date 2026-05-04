# Ranking

To estimate the likelihood that a Text-2-Motion model has encountered certain action patterns we:
1. Take CLIP text encoder
2. Sample one caption per motion from HumanML3D and encode it.
3. Convert action classes into natural language descriptions and encode them also.
4. Compute **crowding distance** as the k-th nearest neighbor in terms of cosine similarity given some NTU action and all HML descriptions.

This approach provides an overall sense of the *semantic alignment* between the two datasets. We interpret this alignment as a proxy for how well the Text-To-Motion model is expected to perform on a given NTU class. A higher crowding indicates a greater concentration of semantically related HumanML3D samples near the NTU reference, suggesting that the model is more likely to generate relevant motions for that category.

## Usage

Before executing, make sure you have text data!
```bash
python3 -m scripts.eda.semantic.rank
```
