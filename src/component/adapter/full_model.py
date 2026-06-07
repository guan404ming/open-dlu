"""Full-model adapter: every parameter is trainable (no layer restriction)."""

from src.component.adapter.base import Adapter


class FullModel(Adapter):
    def setup(self, model):
        for p in model.parameters():
            p.requires_grad = True

    def state_dict(self, model):
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}
