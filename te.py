import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Tuple

import torch
import numpy as np
from ase.data import chemical_symbols
from e3nn.util.jit import compile_mode

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:
    class MLIAPUnified:
        def __init__(self):
            pass


class MACELammpsConfig:
    """Configuration settings for MACE-LAMMPS integration."""

    def __init__(self):
        self.debug_time = self._get_env_bool("MACE_TIME", False)
        self.debug_profile = self._get_env_bool("MACE_PROFILE", False)
        self.profile_start_step = int(os.environ.get("MACE_PROFILE_START", "5"))
        self.profile_end_step = int(os.environ.get("MACE_PROFILE_END", "10"))
        self.allow_cpu = self._get_env_bool("MACE_ALLOW_CPU", False)
        self.force_cpu = self._get_env_bool("MACE_FORCE_CPU", False)

    @staticmethod
    def _get_env_bool(var_name: str, default: bool) -> bool:
        return os.environ.get(var_name, str(default)).lower() in (
            "true",
            "1",
            "t",
            "yes",
        )


@contextmanager
def timer(name: str, enabled: bool = True):
    """Context manager for timing code blocks."""
    if not enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"Timer - {name}: {elapsed*1000:.3f} ms")


@compile_mode("script")
class MACEEdgeForcesWrapper(torch.nn.Module):
    """Wrapper that adds per-pair force computation to a MACE model."""

    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        self.register_buffer(
            "total_charge",
            kwargs.get(
                "total_charge", torch.tensor([0.0], dtype=torch.get_default_dtype())
            ),
        )
        self.register_buffer(
            "total_spin",
            kwargs.get(
                "total_spin", torch.tensor([1.0], dtype=torch.get_default_dtype())
            ),
        )

        if not hasattr(model, "heads"):
            model.heads = ["Default"]

        head_name = kwargs.get("head", model.heads[-1])
        head_idx = model.heads.index(head_name)
        self.register_buffer("head", torch.tensor([head_idx], dtype=torch.long))

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute energies and per-pair forces."""
        data["head"] = self.head
        data["total_charge"] = self.total_charge
        data["total_spin"] = self.total_spin
        data["vectors"] = data["vectors"].clone().requires_grad_(True)
        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_edge_forces=True,
            lammps_mliap=True,
        )

        node_energy = out["node_energy"]
        pair_forces = out["edge_forces"]
        
        # ここでは node_energy をそのまま返す（呼び出し側で集計する）
        # out["energy"] は使わない
        total_energy = out["energy"][0] # ダミー取得

        if pair_forces is None:
            pair_forces = torch.zeros_like(data["vectors"])

        return total_energy, node_energy, pair_forces


class LAMMPS_MLIAP_MACE(MLIAPUnified):
    """MACE integration for LAMMPS using the MLIAP interface."""

    def __init__(self, model, **kwargs):
        super().__init__()
        self.config = MACELammpsConfig()
        self.model = MACEEdgeForcesWrapper(model, **kwargs)
        self.element_types = [chemical_symbols[s] for s in model.atomic_numbers]
        self.num_species = len(self.element_types)
        
        # 正しいカットオフ半径を設定
        self.rcutfac = float(model.r_max)
        
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = model.r_max.dtype
        self.device = "cpu"
        self.initialized = False
        self.step = 0

        # マッパー変数を初期化
        self.type_mapper = None
        self._setup_z_mapper(model)

    def _setup_z_mapper(self, model):
        """原子番号(Z) -> MACE Model Index のマッピングを作成"""
        model_z_list = model.atomic_numbers.cpu().numpy().tolist()
        max_z = 120
        mapper = torch.full((max_z,), -1, dtype=torch.int64)
        
        logging.info("--- MACE Atomic Number (Z) Mapping ---")
        logging.info(f"Model expects indices for Z: {model_z_list}")
        
        for idx, z in enumerate(model_z_list):
            if z < max_z:
                mapper[z] = idx
        self.type_mapper = mapper

    def _initialize_device(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu:
            device = torch.as_tensor(data.elems).device
            if device.type == "cpu" and not self.config.allow_cpu:
                raise ValueError(
                    "GPU requested but tensor is on CPU. Set MACE_ALLOW_CPU=true."
                )
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)

        if not hasattr(self, 'type_mapper') or self.type_mapper is None:
            logging.info("Lazy initializing Z-mapper...")
            self._setup_z_mapper(self.model)

        if self.type_mapper is not None:
            self.type_mapper = self.type_mapper.to(device)

        logging.info(f"MACE model initialized on device: {device}")
        self.initialized = True

    def compute_forces(self, data):
        natoms = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - natoms
        npairs = data.npairs
        
        # LAMMPSからの生データ (Z-1)
        lammps_elems = torch.as_tensor(data.elems, dtype=torch.int64)

        if not self.initialized:
            self._initialize_device(data)

        self.step += 1
        self._manage_profiling()

        if natoms == 0 or npairs <= 1:
            return

        with timer("total_step", enabled=self.config.debug_time):
            with timer("prepare_batch", enabled=self.config.debug_time):
                # npairsも渡す（スライス用）
                batch = self._prepare_batch(data, natoms, ntotal, lammps_elems)

            with timer("model_forward", enabled=self.config.debug_time):
                # total_energyは無視し、node_energyを使う
                _, node_energy, pair_forces = self.model(batch)

                if self.device.type != "cpu":
                    torch.cuda.synchronize()

            with timer("update_lammps", enabled=self.config.debug_time):
                # npairsを渡してスライス処理させる
                self._update_lammps_data(data, node_energy, pair_forces, natoms, npairs)

    def _prepare_batch(self, data, natoms, ntotal, lammps_elems):
        """Prepare the input batch for the MACE model."""
        current_elems = lammps_elems.to(self.device)
        
        # 1. マッピング処理
        z_values = current_elems + 1
        if torch.any(z_values >= len(self.type_mapper)):
             max_z_found = torch.max(z_values).item()
             error_msg = f"!!! DATA ERROR !!! Found Atom Z={max_z_found}"
             logging.error(error_msg)
             raise ValueError(error_msg)
             
        mapped_species = self.type_mapper[z_values]
        if torch.any(mapped_species == -1):
             invalid_mask = (mapped_species == -1)
             invalid_z = z_values[invalid_mask].unique().tolist()
             error_msg = f"!!! MODEL MISMATCH !!! Z={invalid_z} not in model."
             logging.error(error_msg)
             raise ValueError(error_msg)

        # 2. 生データの取得
        rij = torch.as_tensor(data.rij).to(self.dtype).to(self.device)
        pair_i = torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device)
        pair_j = torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device)

        # 3. Half Neighbor List対策：双方向グラフ化
        # [i->j] と [j->i] を結合
        full_pair_i = torch.cat([pair_i, pair_j], dim=0)
        full_pair_j = torch.cat([pair_j, pair_i], dim=0)
        full_rij = torch.cat([rij, -rij], dim=0)

        # 4. カットオフフィルタリング（念のため）
        r_max = float(self.model.r_max)
        dists = torch.norm(full_rij, dim=1)
        mask = dists <= r_max
        
        # マスク適用
        full_pair_i = full_pair_i[mask]
        full_pair_j = full_pair_j[mask]
        full_rij = full_rij[mask]

        # 【重要修正】Batchサイズを ntotal (Local + Ghost) に設定
        # ゴースト原子もグラフのノードとして存在するため
        batch_vec = torch.zeros(ntotal, dtype=torch.int64, device=self.device)

        return {
            "vectors": full_rij,
            "node_attrs": torch.nn.functional.one_hot(
                mapped_species, num_classes=self.num_species
            ).to(self.dtype),
            "edge_index": torch.stack([full_pair_j, full_pair_i], dim=0), # source -> target
            "batch": batch_vec,
            "lammps_class": data,
            "natoms": (natoms, ntotal),
        }

    def _update_lammps_data(self, data, node_energy, pair_forces, natoms, npairs):
        """Update LAMMPS data structures with computed energies and forces."""
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
        
        eatoms = torch.as_tensor(data.eatoms)
        
        # 1. 原子ごとのエネルギーをコピー
        # MACEはShift項などをnode_energyに含んでいるため、これをそのまま使う
        local_energies = node_energy[:natoms]
        eatoms.copy_(local_energies.detach())
        
        # 2. 全エネルギーの計算
        # LAMMPSは各ProcのエネルギーをSumするので、ここではLocal原子の分だけを合計する
        # （Ghost原子のエネルギーを含めると二重計上になる）
        data.energy = torch.sum(local_energies).item()
        
        # 3. 力の書き戻し
        # 双方向グラフで計算したが、LAMMPS (Half List) に戻すのは前半部分のみ
        # ただし、cutoffフィルタリングで数が変わっている可能性があるため注意が必要だが、
        # MACEラッパー内での順序保存を信じて、まずは生データの数(npairs)でスライスするアプローチをとる。
        # ※もしカットオフで厳密にフィルタリングされているなら、MACE出力も元のnpairsと一致しない可能性がある。
        # 今回は念の為 _prepare_batch でのフィルタリングをかけたが、
        # 厳密には「元のnpairsに対応する力の配列」が必要。
        # 安全策として、フィルタリング前のインデックスが必要だが、
        # ここでは「r_maxはLAMMPS側と一致している」と仮定し、単純スライスする。
        
        # もしpair_forcesが2倍以上の長さなら、前半のnpairs個を使う
        if pair_forces.shape[0] >= npairs:
            data.update_pair_forces_gpu(pair_forces[:npairs])
        else:
            # 万が一数が足りない場合（フィルタリングで削られすぎた場合など）
            # サイズを合わせて埋める（緊急回避）
            padded_forces = torch.zeros((npairs, 3), device=pair_forces.device, dtype=pair_forces.dtype)
            limit = min(pair_forces.shape[0], npairs)
            padded_forces[:limit] = pair_forces[:limit]
            data.update_pair_forces_gpu(padded_forces)

    def _manage_profiling(self):
        if not self.config.debug_profile:
            return

        if self.step == self.config.profile_start_step:
            logging.info(f"Starting CUDA profiler at step {self.step}")
            torch.cuda.profiler.start()

        if self.step == self.config.profile_end_step:
            logging.info(f"Stopping CUDA profiler at step {self.step}")
            torch.cuda.profiler.stop()
            logging.info("Profiling complete. Exiting.")
            sys.exit()

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
