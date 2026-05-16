import os
import argparse
import time
import torch
from typing import Dict

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Import ExoSpecMiner components
from exospecminer.phase5_orchestration.ray_slurm import RaySlurmClusterManager, launch_distributed_training


def parse_args():
    parser = argparse.ArgumentParser(description="ExoSpecMiner Distributed HPC Workflow Orchestrator")
    parser.add_argument("--head-ip", type=str, default=None, help="Ray head node IP address")
    parser.add_argument("--port", type=int, default=6379, help="Ray redis port")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of distributed Ray workers")
    parser.add_argument("--num-gpus-per-worker", type=int, default=1, help="GPUs allocated per Ray worker")
    parser.add_argument("--output-dir", type=str, default="hpc_production_run", help="Directory for saving production models and logs")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("    EXOSPECMINER PRODUCTION HPC DISTRIBUTED WORKFLOW    ")
    print("="*60)

    # 1. Initialize Ray Cluster Connection
    cluster_manager = RaySlurmClusterManager(head_node_ip=args.head_ip, redis_port=args.port)
    cluster_manager.initialize_cluster(num_gpus_per_node=args.num_gpus_per_worker * args.num_workers)

    print(f"\n[HPC Cluster] Ray Cluster successfully connected. Resources: {ray.cluster_resources() if RAY_AVAILABLE else 'Local Simulation'}")

    # 2. Define Production Training Configuration
    production_config = {
        "workflow_id": f"exospecminer_prod_{int(time.time())}",
        "output_dir": args.output_dir,
        "phase1_pretrain_epochs": 100,
        "phase2_bae_loops": 10,
        "phase3_cnf_epochs": 50,
        "phase4_domain_epochs": 50,
        "batch_size": 256,
        "learning_rate": 3e-4,
    }

    print(f"\n[HPC Cluster] Launching distributed training across {args.num_workers} Ray workers...")
    start_time = time.time()

    # Launch distributed training
    results = launch_distributed_training(production_config, num_workers=args.num_workers)

    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print(f" [SUCCESS] PRODUCTION HPC DISTRIBUTED WORKFLOW COMPLETED IN {elapsed_time/3600:.2f} HOURS ")
    print(f" Outputs, pre-trained weights, and logs saved to: {args.output_dir} ")
    print("="*60)

    # Shutdown cluster connection
    cluster_manager.shutdown_cluster()


if __name__ == "__main__":
    main()
