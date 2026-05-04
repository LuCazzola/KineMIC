
from torch import nn

class Classifier(nn.Module):
    def __init__(self, *, model, in_dim, num_classes, dropout=0.1):
        super(Classifier, self).__init__()
        self.model = model
        self.drop = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.cls_head = nn.Linear(in_dim, num_classes)

    def forward(self, x, *args):
        x = self.model(x, *args) if args and args[0] is not None else self.model(x)
        if x.dim() > 2:
            x = x.mean(dim=[-1, -2]) # Global Average Pooling
        feats = x.clone()
        x = self.drop(x)
        logits = self.cls_head(x)

        return feats, logits