import torch
import torch.nn.functional as F
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.parallel_1d._utils import gather_forward_split_backward
from torch import Tensor, nn


class Embedding1D(nn.Module):
    """
    1D Embedding.

    Args:
        num_embeddings (int): The size of vocab.
        embedding_dim (int): The dimention of model.
        padding_idx (int): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                            therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                            i.e. it remains as a fixed "pad". None by default.
        dtype (Optional[torch.dtype]): Data type None by default.

    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *args,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        embed_dim_per_partition = embedding_dim // gpc.tensor_parallel_size

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = nn.Parameter(torch.empty((num_embeddings, embed_dim_per_partition), dtype=dtype))

    def forward(self, input_: Tensor) -> Tensor:

        output_parallel = F.embedding(input_, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        output = gather_forward_split_backward(output_parallel, ParallelMode.PARALLEL_1D, dim=-1)

        return output
