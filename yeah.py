"""
MACE-OMAT-0 Foundation Model を使った Si 原子の MD ベンチマーク
原子数: 100, 1000, 10000, 20000, 30000, 40000, 50000
MD steps: 200
"""

import time
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from mace.calculators import MACECalculator
import torch

def create_si_supercell(target_atoms: int) -> Atoms:
    """
    指定した原子数に近いSi supercellを作成
    Si diamond構造: 単位胞あたり8原子
    """
    si_unit = bulk('Si', 'diamond', a=5.43)
    
    # 必要な単位胞数
    n_cells_needed = target_atoms / 8
    
    # 立方体に近い形で探索（探索範囲を十分に広げる）
    n_max = int(np.ceil(n_cells_needed ** (1/3))) + 10
    
    best_repeat = (1, 1, 1)
    best_diff = float('inf')
    
    # 全探索
    for nx in range(1, n_max + 1):
        for ny in range(1, n_max + 1):
            for nz in range(1, n_max + 1):
                n_atoms = 8 * nx * ny * nz
                diff = abs(n_atoms - target_atoms)
                
                # より近いものを見つけたら更新
                if diff < best_diff:
                    best_diff = diff
                    best_repeat = (nx, ny, nz)
                
                # 完全一致なら終了
                if diff == 0:
                    supercell = si_unit.repeat(best_repeat)
                    return supercell
                
                # これ以上nzを増やしても離れるだけなので打ち切り
                if n_atoms > target_atoms:
                    break
    
    supercell = si_unit.repeat(best_repeat)
    return supercell


def run_md_benchmark(atoms: Atoms, calculator, n_steps: int = 200, 
                     temperature: float = 300.0, timestep: float = 1.0) -> dict:
    """
    MDを実行してベンチマーク結果を返す
    """
    atoms.calc = calculator
    
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = VelocityVerlet(atoms, timestep * units.fs)
    
    # ウォームアップ
    warmup_steps = 5
    for _ in range(warmup_steps):
        dyn.run(1)
    
    # GPU同期してから計測開始
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    dyn.run(n_steps)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    time_per_step = elapsed_time / n_steps
    
    return {
        'n_atoms': len(atoms),
        'n_steps': n_steps,
        'total_time_sec': elapsed_time,
        'time_per_step_sec': time_per_step,
        'steps_per_sec': n_steps / elapsed_time
    }


def main():
    target_atom_counts = [100, 1000, 10000, 20000, 30000, 40000, 50000]
    n_steps = 200
    temperature = 300.0
    timestep = 1.0
    
    # デバイス確認
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"\nLoading MACE-OMAT-0 model on {device}...")
    
    calculator = MACECalculator(
        model_path="mace-omat-0-medium.model",
        device=device,
        default_dtype="float64"
    )
    
    print(f"Model loaded successfully!")
    print(f"MD steps: {n_steps}")
    print(f"Temperature: {temperature} K")
    print(f"Timestep: {timestep} fs")
    print("=" * 70)
    
    # 事前に原子数を確認表示
    print("\nTarget vs Actual atom counts:")
    for target_n in target_atom_counts:
        atoms = create_si_supercell(target_n)
        print(f"  Target: {target_n:>6} -> Actual: {len(atoms):>6}")
    print("=" * 70)
    
    results = []
    
    for target_n in target_atom_counts:
        print(f"\nPreparing Si supercell with ~{target_n} atoms...")
        atoms = create_si_supercell(target_n)
        actual_n = len(atoms)
        print(f"Actual atom count: {actual_n}")
        
        # メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Running MD benchmark ({n_steps} steps)...")
        try:
            result = run_md_benchmark(
                atoms.copy(), 
                calculator, 
                n_steps=n_steps,
                temperature=temperature,
                timestep=timestep
            )
            results.append(result)
            
            print(f"  Total time: {result['total_time_sec']:.2f} sec")
            print(f"  Time/step: {result['time_per_step_sec']*1000:.2f} ms")
            print(f"  Steps/sec: {result['steps_per_sec']:.2f}")
            
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            if "out of memory" in str(e).lower():
                print("  -> GPU memory insufficient for this system size")
            results.append({
                'n_atoms': actual_n,
                'n_steps': n_steps,
                'total_time_sec': float('nan'),
                'time_per_step_sec': float('nan'),
                'steps_per_sec': float('nan')
            })
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Target':>10} {'Actual':>10} {'Total (s)':>12} {'Per Step (ms)':>15} {'Steps/s':>12}")
    print("-" * 70)
    
    for target, r in zip(target_atom_counts, results):
        print(f"{target:>10} {r['n_atoms']:>10} {r['total_time_sec']:>12.2f} "
              f"{r['time_per_step_sec']*1000:>15.2f} {r['steps_per_sec']:>12.2f}")
    
    # CSV保存
    csv_filename = "mace_benchmark_results.csv"
    with open(csv_filename, 'w') as f:
        f.write("target_atoms,actual_atoms,n_steps,total_time_sec,time_per_step_sec,steps_per_sec\n")
        for target, r in zip(target_atom_counts, results):
            f.write(f"{target},{r['n_atoms']},{r['n_steps']},{r['total_time_sec']:.4f},"
                    f"{r['time_per_step_sec']:.6f},{r['steps_per_sec']:.4f}\n")
    print(f"\nResults saved to {csv_filename}")


if __name__ == "__main__":
    main()
