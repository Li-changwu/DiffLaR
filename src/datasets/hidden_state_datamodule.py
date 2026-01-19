"""
Hidden State DataModule

用于 Stage1 训练，加载预保存的 Hidden States。
Stage2 训练时可以使用原始的 QSA DataModule。
"""

from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from ..data.hidden_state_dataset import HiddenStateDataset


class HiddenStateDataModule(pl.LightningDataModule):
    """
    Hidden State DataModule
    
    用于 Stage1 训练，加载预保存的 Hidden States。
    """
    
    def __init__(
        self,
        hidden_states_dir: str,
        dataset_name: str = "hidden_states",
        all_config=None,
    ):
        super().__init__()
        self.hidden_states_dir = Path(hidden_states_dir)
        self.dataset_name = dataset_name
        self.all_config = all_config
        self.batch_size = all_config.dataloader.batch_size
        
        self.train_set = None
        self.val_set = None
    
    def setup(self, stage: str = None):
        """设置数据集"""
        if stage == "fit":
            self.train_set = HiddenStateDataset(
                data_dir=str(self.hidden_states_dir),
                split="train"
            )
            self.val_set = HiddenStateDataset(
                data_dir=str(self.hidden_states_dir),
                split="val"
            )
        elif stage == "test":
            self.test_set = HiddenStateDataset(
                data_dir=str(self.hidden_states_dir),
                split="test"
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.all_config.dataloader.get("num_workers", 4),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 32),
            shuffle=False,
            num_workers=self.all_config.dataloader.get("num_workers", 4),
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 32),
            shuffle=False,
            num_workers=self.all_config.dataloader.get("num_workers", 4),
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
        )


