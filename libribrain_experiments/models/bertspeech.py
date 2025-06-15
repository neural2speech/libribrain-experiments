import torch, torch.nn as nn
from transformers import BertModel, BertConfig


class BertSpeech(nn.Module):
    """Model based in DyslexNet, implementation simplified using BERT from HF.

    Each time-step is projected to `emb_size` and then to the *factorised*
    dimension `factor_size` before reaching the full Transformer hidden size.
    All encoder layers **share parameters** (like ALBERT) to keep the model
    small enough for the dataset.

    - Paper: https://sciencedirect.com/science/article/pii/S1053811923002185

    Parameters
    ----------
    input_dim : int
        Number of sensors (features per time-step).
    seq_len : int
        Number of time-steps in the input epoch.
    num_classes : int, default `2`
    hidden_size : int, default `3072`
    emb_size : int, default `768`
    factor_size : int, default `128`
    num_heads : int, default `12`
    num_layers : int, default `4`
        Logical number of layers **before sharing**.
    """

    def __init__(
        self,
        input_dim,
        seq_len=125,
        num_classes=2,
        hidden_size=3072,
        emb_size=768,
        factor_size=128,
        num_heads=12,
        num_layers=4,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_size)
        self.factorization = nn.Linear(emb_size, factor_size)
        self.projection = nn.Linear(factor_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, hidden_size))

        config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=seq_len,
            vocab_size=1,  # dummy
            pad_token_id=0,
        )
        self.encoder = BertModel(config)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """Define the computation performed at every call.

        Parameters
        ----------
        x : torch.Tensor, shape `(batch, channels, time)`
            Raw MEG epoch.

        Returns
        -------
        torch.Tensor
            Logits of shape `(batch, num_classes)`.
        """
        # from (batch, channels, timesteps) to (batch, timesteps, channels)
        x = x.transpose(1, 2)

        x = self.embedding(x)
        x = self.factorization(x)
        x = self.projection(x)
        x = x + self.pos_encoding
        outputs = self.encoder(inputs_embeds=x)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled_output)
