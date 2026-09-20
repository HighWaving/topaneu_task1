from .anet_roi_net import (
    AneurysmRoiBackboneNnUNetTruncatedDecoder,
    apply_chunked_conv3d_to_highres,
    chunked_conv3d,
)
from .topaneu_vessel_aware import TopAneuVesselAwareClassifier
from .topaneu_dilation import TopAneuVesselAwareMeanPoolDilationR1
from .topaneu_listwise import TopAneuVesselAwareListwise
from .topaneu_loctoken import TopAneuLocationTokenClassifier
from .topaneu_dense import TopAneuDenseSupervised

MODEL_CLASSES = {
    "TopAneuDenseSupervised": TopAneuDenseSupervised,
    "TopAneuVesselAwareClassifier": TopAneuVesselAwareClassifier,
    "TopAneuVesselAwareMeanPoolDilationR1": TopAneuVesselAwareMeanPoolDilationR1,
    "TopAneuVesselAwareListwise": TopAneuVesselAwareListwise,
    "TopAneuLocationTokenClassifier": TopAneuLocationTokenClassifier,
}

__all__ = [
    "AneurysmRoiBackboneNnUNetTruncatedDecoder",
    "TopAneuVesselAwareClassifier",
    "TopAneuVesselAwareMeanPoolDilationR1",
    "TopAneuVesselAwareListwise",
    "TopAneuLocationTokenClassifier",
    "TopAneuDenseSupervised",
    "MODEL_CLASSES",
    "chunked_conv3d",
    "apply_chunked_conv3d_to_highres",
]
