#!/usr/bin/env python3
import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ENsiRNA_mod.data.dataset import E2EDataset
from ENsiRNA_mod.model.mask_model import RNAmaskModel
from ENsiRNA_mod.utils.random_seed import setup_seed


class ENSIRNAModWrapper:
    def __init__(
        self,
        ckpt_path: str = "ENsiRNA-mod/pkl/checkpoint_1.ckpt",
        gpu: int = -1,
    ):
        self.ckpt_path = ckpt_path
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
        sense_mod: Dict[str, str] = None,
        anti_mod: Dict[str, str] = None,
        save_dir: str = "results",
    ) -> Dict[str, Any]:
        self.load_model()

        os.makedirs(save_dir, exist_ok=True)

        if sense_mod is None:
            sense_mod = {}
        if anti_mod is None:
            anti_mod = ""

        sense_mod_str = " * ".join(list(sense_mod.keys())) if sense_mod else ""
        sense_pos_str = "* ".join(list(sense_mod.values())) if sense_mod else ""

        anti_mod_str = " * ".join(list(anti_mod.keys())) if anti_mod else ""
        anti_pos_str = "* ".join(list(anti_mod.values())) if anti_mod else ""

        test_item = {
            "ID": siRNA_id,
            "source": 0,
            "cc": 0,
            "sense raw seq": sense_seq.upper(),
            "sense mod": sense_mod_str,
            "sense pos": sense_pos_str,
            "anti raw seq": anti_seq.upper(),
            "anti mod": anti_mod_str,
            "anti pos": anti_pos_str,
            "PCT": 0,
            "anti length": len(anti_seq),
            "sense length": len(sense_seq),
            "cc_norm": 0,
            "group": 0,
            "pdb_data_path": "",
            "start": 30,
            "chain": 3,
            "atom_mask": [],
            "smask": list(range(1, len(sense_seq) + len(anti_seq) + 1)),
            "sense seq": sense_seq.upper() + anti_seq.upper(),
        }

        temp_json_path = os.path.join(save_dir, f"{siRNA_id}_mod_temp.json")
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

                prob, _ = self.model.test(**batch)
                result = prob.cpu().tolist()[0] if len(prob) > 0 else 0.0

            os.remove(temp_json_path)

            return {
                "id": siRNA_id,
                "sense_seq": sense_seq,
                "anti_seq": anti_seq,
                "sense_mod": sense_mod,
                "anti_mod": anti_mod,
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
                "sense_mod": sense_mod,
                "anti_mod": anti_mod,
                "result": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


def predict_ensiRNA_mod(
    siRNA_id: str,
    sense_seq: str,
    anti_seq: str,
    sense_mod: Dict[str, str] = None,
    anti_mod: Dict[str, str] = None,
    ckpt_path: str = "ENsiRNA-mod/pkl/checkpoint_1.ckpt",
    gpu: int = -1,
) -> Dict[str, Any]:
    wrapper = ENSIRNAModWrapper(ckpt_path=ckpt_path, gpu=gpu)
    return wrapper.predict(siRNA_id, sense_seq, anti_seq, sense_mod, anti_mod)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ENsiRNA-Mod prediction wrapper")
    parser.add_argument("--id", type=str, required=True, help="siRNA ID")
    parser.add_argument("--sense-seq", type=str, required=True, help="Sense sequence")
    parser.add_argument(
        "--anti-seq", type=str, required=True, help="Anti-sense sequence"
    )
    parser.add_argument(
        "--sense-mod",
        type=str,
        default="",
        help="Sense modification (format: type:pos,type:pos)",
    )
    parser.add_argument(
        "--anti-mod",
        type=str,
        default="",
        help="Anti-sense modification (format: type:pos,type:pos)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ENsiRNA-mod/pkl/checkpoint_1.ckpt",
        help="Checkpoint path",
    )
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device")
    args = parser.parse_args()

    def parse_mod(mod_str):
        if not mod_str:
            return {}
        mod_dict = {}
        for item in mod_str.split(","):
            if ":" in item:
                mod_type, mod_pos = item.split(":")
                mod_dict[mod_type] = mod_pos
        return mod_dict

    result = predict_ensiRNA_mod(
        args.id,
        args.sense_seq,
        args.anti_seq,
        parse_mod(args.sense_mod),
        parse_mod(args.anti_mod),
        args.ckpt,
        args.gpu,
    )
    print(json.dumps(result, indent=2))
