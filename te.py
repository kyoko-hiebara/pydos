import logging
import os
import sys
import time
import ctypes
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
    def __init__(self):
        self.debug_time = self._get_env_bool("MACE_TIME", False)
        self.debug_profile = self._get_env_bool("MACE_PROFILE", False)
        # 環境変数からデバイス指定を取得 (未指定なら空文字)
        self.device_str = os.environ.get("MACE_DEVICE", "") 

    @staticmethod
    def _get_env_bool(var_name: str, default: bool) -> bool:
        return os.environ.get(var_name, str(default)).lower() in ("true", "1", "t", "yes")

@contextmanager
def timer(name: str, enabled: bool = True):
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
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        self.register_buffer("total_charge", kwargs.get("total_charge", torch.tensor([0.0])))
        self.register_buffer("total_spin", kwargs.get("total_spin", torch.tensor([1.0])))

        if not hasattr(model, "heads"):
            model.heads = ["Default"]

        head_name = kwargs.get("head", model.heads[-1])
        head_idx = model.heads.index(head_name)
        self.register_buffer("head", torch.tensor([head_idx], dtype=torch.long))

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return out["energy"][0], out["node_energy"], out["edge_forces"]

class LAMMPS_MLIAP_MACE(MLIAPUnified):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.config = MACELammpsConfig()
        self.model = MACEEdgeForcesWrapper(model, **kwargs)
        self.element_types = [chemical_symbols[s] for s in model.atomic_numbers]
        self.num_species = len(self.element_types)
        
        self.rcutfac = float(model.r_max)
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = model.r_max.dtype
        self.device = "cpu"
        self.initialized = False
        self.step = 0
        
        self.type_mapper = None
        self._setup_z_mapper(model)

    def _setup_z_mapper(self, model):
        model_z_list = model.atomic_numbers.cpu().numpy().tolist()
        max_z = 120
        mapper = torch.full((max_z,), -1, dtype=torch.int64)
        for idx, z in enumerate(model_z_list):
            if z < max_z:
                mapper[z] = idx
        self.type_mapper = mapper

    def _initialize_device(self, data):
        # Configの安全確保
        if not hasattr(self, 'config'):
             self.config = MACELammpsConfig()

        device_str = getattr(self.config, "device_str", "")
        if device_str:
            device = torch.device(device_str)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.device = device
        self.model = self.model.to(device)

        # Mapperの自己修復
        if not hasattr(self, 'type_mapper') or self.type_mapper is None:
            self._setup_z_mapper(self.model)

        self.type_mapper = self.type_mapper.to(device)
        logging.info(f"MACE model initialized on device: {device}")
        self.initialized = True

    def _get_all_atom_types_hack(self, data_elems, ntotal):
        """
        [Ultra Memory Hack]
        data.elems (Python object) からメモリアドレスを特定し、
        Local境界を突破してGhost原子を含む ntotal 個のデータを強制的に読み出す。
        """
        # アドレス取得を試みる（複数のプロトコルに対応）
        addr = None
        
        # 1. Array Interface (Numpy/Cython)
        if hasattr(data_elems, '__array_interface__'):
             try:
                 addr = data_elems.__array_interface__['data'][0]
             except Exception:
                 pass
                 
        # 2. Ctypes (C-types wrapping)
        if addr is None and hasattr(data_elems, 'ctypes'):
             try:
                 addr = data_elems.ctypes.data
             except Exception:
                 pass

        # 3. Buffer Protocol (MemoryView) -> Ctypes cast
        if addr is None:
             try:
                 mv = memoryview(data_elems)
                 # バッファの先頭アドレスを取得するCtypes魔術
                 addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
             except Exception:
                 pass
        
        if addr is None:
             raise ValueError("Could not determine memory address of atom types.")

        # LAMMPSの atom->type は通常 C int (32bit)
        # アドレスから ntotal 個の int 配列としてマッピングする
        try:
            # int型のポインタとして解釈
            c_int_p = ctypes.POINTER(ctypes.c_int)
            ptr = ctypes.cast(addr, c_int_p)
            
            # Numpy配列としてビューを作成 (コピーを作成してTorchへ)
            # ここで ntotal サイズを指定することで、Local外のメモリも読む
            full_array = np.ctypeslib.as_array(ptr, shape=(ntotal,))
            return torch.from_numpy(full_array.copy()).to(torch.int64)
            
        except Exception as e:
            raise ValueError(f"Memory hack execution failed: {e}")

    def compute_forces(self, data):
        natoms = data.nlocal
        ntotal = data.ntotal
        npairs = data.npairs

        if not self.initialized:
            self._initialize_device(data)

        # --- [Step 1] Ghost原子読み出しハック ---
        try:
            lammps_elems = self._get_all_atom_types_hack(data.elems, ntotal)
            # 成功ログ (デバッグ用)
            # logging.info(f"Hack Success: Retrieved {lammps_elems.size(0)} atoms (Target: {ntotal})")
        except ValueError as e:
            logging.error(f"Memory Hack Failed: {e}")
            # フォールバック: そのまま使う（恐らく33個）
            lammps_elems = torch.as_tensor(data.elems, dtype=torch.int64)

        # --- [Step 2] 致命的エラーのチェック ---
        # ここで数が足りなければ MACE は 100% クラッシュするので、
        # 親切なエラーメッセージを出して Python 側で止める。
        if lammps_elems.size(0) < ntotal:
             error_msg = (
                 f"!!! CRITICAL ERROR !!!\n"
                 f"MACE needs {ntotal} atoms (Local+Ghost), but LAMMPS provided only {lammps_elems.size(0)}.\n"
                 f"The memory hack failed to retrieve ghost atoms.\n"
                 f"Troubleshooting:\n"
                 f"1. Ensure you are NOT using '-sf kk' or '-pk kokkos' (CPU mode is required for this hack).\n"
                 f"2. Your LAMMPS python wrapper might be strictly safe-guarded."
             )
             logging.error(error_msg)
             raise RuntimeError(error_msg)

        self.step += 1
        self._manage_profiling()
        
        if natoms == 0 or npairs <= 1:
            return

        with timer("MACE_Step", enabled=self.config.debug_time):
            # Batch準備
            batch = self._prepare_batch(data, natoms, ntotal, lammps_elems)
            
            # MACE計算
            _, node_energy, pair_forces = self.model(batch)
            
            if self.device.type != "cpu":
                torch.cuda.synchronize()

            # 書き戻し
            self._update_lammps_data(data, node_energy, pair_forces, natoms, npairs)

    def _prepare_batch(self, data, natoms, ntotal, lammps_elems):
        current_elems = lammps_elems.to(self.device)
        
        # マッピング
        z_values = current_elems + 1
        
        # マッピングエラーチェック
        if torch.any(z_values >= len(self.type_mapper)):
             max_z = torch.max(z_values).item()
             raise ValueError(f"Found undefined atom Z={max_z}")
             
        mapped_species = self.type_mapper[z_values]
        if torch.any(mapped_species == -1):
             raise ValueError("Found atom Z not in model support list.")

        # 座標とペアリスト
        rij = torch.as_tensor(data.rij).to(self.dtype).to(self.device)
        pair_i = torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device)
        pair_j = torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device)

        # 双方向グラフ化
        full_pair_i = torch.cat([pair_i, pair_j], dim=0)
        full_pair_j = torch.cat([pair_j, pair_i], dim=0)
        full_rij = torch.cat([rij, -rij], dim=0)
        
        # カットオフフィルタ
        dists = torch.norm(full_rij, dim=1)
        r_max = float(self.model.r_max)
        mask = dists <= r_max
        
        full_pair_i = full_pair_i[mask]
        full_pair_j = full_pair_j[mask]
        full_rij = full_rij[mask]

        # Batchサイズ
        batch_vec = torch.zeros(ntotal, dtype=torch.int64, device=self.device)

        return {
            "vectors": full_rij,
            "node_attrs": torch.nn.functional.one_hot(mapped_species, num_classes=self.num_species).to(self.dtype),
            "edge_index": torch.stack([full_pair_j, full_pair_i], dim=0),
            "batch": batch_vec,
            "lammps_class": data,
            "natoms": (natoms, ntotal),
        }

    def _update_lammps_data(self, data, node_energy, pair_forces, natoms, npairs):
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
            
        eatoms = torch.as_tensor(data.eatoms)
        
        # エネルギー
        eatoms.copy_(node_energy[:natoms].detach())
        data.energy = torch.sum(node_energy[:natoms]).item()
        
        # 力
        if pair_forces.shape[0] >= npairs:
            data.update_pair_forces(pair_forces[:npairs].cpu().numpy())
        else:
            padded = torch.zeros((npairs, 3), device=pair_forces.device, dtype=pair_forces.dtype)
            limit = pair_forces.shape[0]
            padded[:limit] = pair_forces
            data.update_pair_forces(padded.cpu().numpy())

    def _manage_profiling(self):
        if not self.config.debug_profile:
            return
        if self.step == self.config.profile_start_step:
            torch.cuda.profiler.start()
        if self.step == self.config.profile_end_step:
            torch.cuda.profiler.stop()
            sys.exit()

    def compute_descriptors(self, data): pass
    def compute_gradients(self, data): pass
