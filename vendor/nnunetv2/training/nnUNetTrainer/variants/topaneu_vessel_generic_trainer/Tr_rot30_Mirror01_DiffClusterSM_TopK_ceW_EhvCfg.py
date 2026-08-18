"""Compatibility name for a historical TA36 checkpoint trainer typo.

The Official TA36 implementation is ``..._EnvCfg``. Some exported checkpoint
metadata spells the same trainer ``..._EhvCfg``. Keep trainer discovery
compatible without duplicating or modifying the organizer's implementation.
"""

from nnunetv2.training.nnUNetTrainer.variants.topaneu_vessel_generic_trainer.Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg import (
    Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg,
)


class Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EhvCfg(
    Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg
):
    """Alias subclass; all trainer behavior comes from Official TA36 EnvCfg."""

