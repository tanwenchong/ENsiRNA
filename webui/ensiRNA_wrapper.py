#!/usr/bin/env python3
import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ENsiRNA.data.dataset import E2EDataset
from ENsiRNA.model.mask_model import RNAmaskModel
from ENsiRNA.utils.random_seed import setup_seed


class ENSIRNAWrapper:
    def __init__(
        self,
        ckpt_path: str = "ENsiRNA/pkl/checkpoint_1.ckpt",
        model_config: str = "ENsiRNA/config.json",
        gpu: int = -1,
    ):
        self.ckpt_path = ckpt_path
        self.model_config = model_config
        self.gpu = gpu
        self.model = None
        self.device = None

    def load_model(self):
        if self.model is None:
            self.model = torch.load(
                self.ckpt_path, map_location="cpu", weights_only=False
            )
            self.device = torch.device("cpu" if self.gpu == -1 else f"cuda:{self.gpu}")
            self.model.to(self.device)
            self.model.eval()
        return self.model

    def predict(
        self,
        siRNA_id: str,
        sense_seq: str,
        anti_seq: str,
        mRNA_seq: str = "",
        position: int = 30,
        save_dir: str = "results",
    ) -> Dict[str, Any]:
        self.load_model()

        os.makedirs(save_dir, exist_ok=True)

        test_item = {
            "siRNA": siRNA_id,
            "sense seq": sense_seq.upper(),
            "anti seq": anti_seq.upper(),
            "mRNA_seq": mRNA_seq if mRNA_seq else "A" * 61,
            "position": position,
            "pdb_data_path": "",
            "start": 30,
            "chain": 3,
        }

        temp_json_path = os.path.join(save_dir, f"{siRNA_id}_temp.json")
        with open(temp_json_path, "w") as f:
            f.write(json.dumps(test_item) + "\n")

        try:
            test_set = E2EDataset(temp_json_path)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                num_workers=0,
                collate_fn=E2EDataset.collate_fn,
                shuffle=False,
            )

            result = None
            for batch in test_loader:
                for k in batch:
                    if hasattr(batch[k], "to"):
                        batch[k] = batch[k].to(self.device)

                prob, _, _, _ = self.model.test(**batch)
                result = prob.cpu().tolist()[0] if len(prob) > 0 else 0.0

            os.remove(temp_json_path)

            return {
                "id": siRNA_id,
                "sense_seq": sense_seq,
                "anti_seq": anti_seq,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            if os.path.exists(temp_json_path):
                os.remove(temp_json_path)
            return {
                "id": siRNA_id,
                "sense_seq": sense_seq,
                "anti_seq": anti_seq,
                "result": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


def predict_ensiRNA(
    siRNA_id: str,
    sense_seq: str,
    anti_seq: str,
    mRNA_seq: str = "",
    position: int = 30,
    ckpt_path: str = "ENsiRNA/pkl/checkpoint_1.ckpt",
    gpu: int = -1,
) -> Dict[str, Any]:
    wrapper = ENSIRNAWrapper(ckpt_path=ckpt_path, gpu=gpu)
    return wrapper.predict(siRNA_id, sense_seq, anti_seq, mRNA_seq, position)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ENsiRNA prediction wrapper")
    parser.add_argument("--id", type=str, required=True, help="siRNA ID")
    parser.add_argument("--sense-seq", type=str, required=True, help="Sense sequence")
    parser.add_argument(
        "--anti-seq", type=str, required=True, help="Anti-sense sequence"
    )
    parser.add_argument("--mRNA-seq", type=str, default="", help="mRNA sequence")
    parser.add_argument("--position", type=int, default=30, help="Position")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ENsiRNA/pkl/checkpoint_1.ckpt",
        help="Checkpoint path",
    )
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device")
    args = parser.parse_args()

    result = predict_ensiRNA(
        args.id,
        args.sense_seq,
        args.anti_seq,
        args.mRNA_seq,
        args.position,
        args.ckpt,
        args.gpu,
    )
    print(json.dumps(result, indent=2))
