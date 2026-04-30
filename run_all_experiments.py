#!/usr/bin/env python
"""
Run all experiments in one command.
"""
import subprocess
import sys
from pathlib import Path

def run_experiment(script_name, description):
    """Run a single experiment."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            [sys.executable, f'experiments/{script_name}'],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error: {e}")
        return False


def main():
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║          RhythmGym Experiment Suite                  ║
    ║       Adaptive Piano Practice with Safety            ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    experiments = [
        ('compare.py', 'Algorithm Comparison (8 agents)'),
        ('pianoMPC_horizon.py', 'PianoMPC Horizon Sweep (E2)'),
        ('safety_ablation.py', 'Safety Ablation (E3)'),
        ('safety_consistency.py', 'Safety Consistency Diagnosis (A1)'),
        ('suite_robustness.py', 'Suite Robustness (E5)'),
        ('dynamics_mismatch.py', 'Dynamics Mismatch Robustness (E6)'),
        ('scoped_mismatch.py', 'Scoped Model Mismatch (E10)'),
        ('threshold_window_robustness.py', 'Threshold/Window Robustness (E7)'),
    ]

    results = []
    for script, desc in experiments:
        success = run_experiment(script, desc)
        results.append((desc, success))

    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for desc, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8s} {desc}")

    print(f"\n{'='*70}")
    print(f"All results saved to: output/")
    print(f"{'='*70}\n")

    # Post-processing: generate tables and figures
    post_steps = [
        ("scripts/generate_tables.py", "Generate Tables"),
        ("scripts/plot_figures.py", "Generate Figures"),
    ]
    for script, desc in post_steps:
        print(f"[Post] {desc} ...", end=" ", flush=True)
        try:
            subprocess.run(
                [sys.executable, script],
                cwd=Path(__file__).parent,
                check=True,
                capture_output=False
            )
            print("✓")
        except subprocess.CalledProcessError as exc:
            print(f"✗ (error: {exc.returncode})")

if __name__ == '__main__':
    main()
