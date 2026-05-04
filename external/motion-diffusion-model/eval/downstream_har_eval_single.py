import argparse
import os
import glob
import subprocess
import sys
from pathlib import Path

def run_command(command, action_description):
    """
    Executes a command using subprocess and handles potential errors.
    """
    print(f"\n> {action_description}")
    print(f"  Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, text=True, capture_output=False)
    except FileNotFoundError:
        print(f"Error: The command '{command[0]}' was not found.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while {action_description.lower()}.")
        print(f"Return code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(1)

def main(args):
    """
    Orchestrates a SINGLE downstream HAR evaluation run.
    """
    ckpt_path = args.ckpt_path
    
    if not os.path.isfile(ckpt_path):
        print(f"Error: Checkpoint file not found at '{ckpt_path}'")
        sys.exit(1)

    # --- Extract details for folder naming logic ---
    # Assumes structure: .../experiment_name/model_123.pt
    ckpt_name = os.path.basename(ckpt_path)
    model_dir = os.path.dirname(ckpt_path)
    exp_name = os.path.basename(os.path.normpath(model_dir))
    
    # Extract iteration number (e.g., "3000" from "model_3000.pt")
    try:
        n_iter = ckpt_name.replace('model_', '').replace('.pt', '')
        int(n_iter) # Verify it's a number
    except ValueError:
        print(f"Error: Could not extract iteration number from '{ckpt_name}'. Expected format 'model_X.pt'.")
        sys.exit(1)

    print(f"✅ Target Checkpoint: {ckpt_name} (Exp: {exp_name})")

    # =================================================================================
    # STAGE 1: DATA GENERATION (Optional)
    # =================================================================================
    
    if args.gen_data:
        gen_command = [
            sys.executable, '-m', 'sample.synth_dataset_generate',
            '--model_path', ckpt_path,
            '--seed', str(args.gen_seed)
        ]
        action_desc = f"Generating data for [{ckpt_name}] with seed [{args.gen_seed}]"
        run_command(gen_command, action_desc)
    else:
        print(f"\n> Skipping data generation (using existing data for seed {args.gen_seed}).")

    # =================================================================================
    # STAGE 2: LOCATE DATASET
    # =================================================================================
    
    # Construct the expected directory name based on generation logic
    synth_dir_name = f'{exp_name}_{n_iter}_seed{args.gen_seed}'
    
    # Search for the path (matches original script logic)
    search_pattern = os.path.join('./dataset', '*', 'splits', 'synth', '*', '*', synth_dir_name)
    found_dirs = glob.glob(search_pattern)

    if not found_dirs:
        print(f"\nError: Could not find the dataset directory for '{synth_dir_name}'.")
        print(f"Search pattern used: {search_pattern}")
        print("Did you forget to run with --gen_data?")
        sys.exit(1)
    
    synth_dir = found_dirs[0]
    print(f"✅ Found dataset: {synth_dir}")

    # =================================================================================
    # STAGE 3: SINGLE HAR TRAINING RUN
    # =================================================================================
    
    print(f"\n--- Launching HAR training run ---")
    
    train_command = [
        sys.executable, '-m', 'train.train_har',
        '--data_dir', synth_dir,
        '--seed', str(args.train_seed),
        '--mode', args.mode,
        '--data_rep', args.data_rep,
        '--model_har', args.model_har,
        '--pyskl', args.pyskl,
    ]

    # Append optional flags
    if args.augs:
        train_command.append('--augs')
    if args.keep_data:
        train_command.append('--keep_data')
    if args.use_wandb:
        train_command.append('--use_wandb')
        train_command.extend(['--wandb_project', args.wandb_project])

    action_desc = f"Training HAR model (Seed: {args.train_seed}, Augs: {args.augs})"
    run_command(train_command, action_desc)

    print("\n🎉 Single run finished successfully! 🎉")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single HAR evaluation: generate data from ONE checkpoint, then train ONE HAR model."
    )
    
    # --- Target Checkpoint ---
    parser.add_argument(
        '--ckpt_path', 
        type=str, 
        required=True,
        help="Full path to the specific model checkpoint (e.g., './save/my_exp/model_3000.pt')."
    )

    # --- Seeds ---
    parser.add_argument('--gen_seed', type=int, default=10, help='Seed used for synthetic data generation.')
    parser.add_argument('--train_seed', type=int, default=42, help='Seed used for HAR model initialization.')

    # --- HAR Options ---
    parser.add_argument('--mode', type=str, default='mixed', choices=['gt', 'real', 'synth', 'mixed'], help='Training mode.')
    parser.add_argument('--data_rep', type=str, default='j', choices=['b', 'bm', 'j', 'jm'], help='Data representation.')
    parser.add_argument('--model_har', type=str, default='stgcn', choices=['stgcn'], help='HAR model architecture.')
    parser.add_argument('--pyskl', type=str, default='../pyskl', help='Path to PySKL repo.')
    
    # --- Flags ---
    parser.add_argument('--gen_data', action='store_true', help='If set, runs the generation script before training.')
    parser.add_argument('--augs', action='store_true', help='If set, uses augmentations during HAR training.')
    parser.add_argument('--keep_data', action='store_true', help='Keep the temporary PySKL dataset file.')
    
    # --- W&B ---
    parser.add_argument('--use_wandb', action='store_true', help='Enable W&B logging.')
    parser.add_argument('--wandb_project', type=str, default='har_eval_single', help='W&B project name.')
    
    args = parser.parse_args()
    
    main(args)