import argparse
import os
import glob
import subprocess
import sys
from pathlib import Path

def run_command(command, action_description):
    """
    Executes a command using subprocess and handles potential errors.
    
    Args:
        command (list): The command to execute as a list of strings.
        action_description (str): A description of the action being performed for logging.
    """
    print(f"\n> {action_description}")
    print(f"  Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, text=True, capture_output=False)
    except FileNotFoundError:
        print(f"Error: The command '{command[0]}' was not found.")
        print("Please ensure that Python is in your PATH and you are in the correct directory.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while {action_description.lower()}.")
        print(f"Return code: {e.returncode}")
        # The called script should have printed its own error, so we exit.
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(1)


def main(args):
    """
    Main function to orchestrate the downstream HAR evaluation pipeline in a sequential manner.
    """
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found at '{model_dir}'")
        sys.exit(1)

    # --- Find all model checkpoints ---
    checkpoint_paths = sorted(glob.glob(os.path.join(model_dir, 'model_*.pt')))
    if not checkpoint_paths:
        print(f"Error: No checkpoints matching 'model_*.pt' found in '{model_dir}'")
        sys.exit(1)
        
    print(f"✅ Found {len(checkpoint_paths)} checkpoints in '{model_dir}'.")

    # --- Define seeds and options for the loops ---
    gen_seeds = [10]
    train_seeds = [42, 123, 19]
    augmentation_options = [False]
    exp_name = os.path.basename(os.path.normpath(model_dir))

    # =================================================================================
    # SEQUENTIAL GENERATION AND TRAINING LOOP
    # =================================================================================
    
    # Outermost loop: Generation Seed
    for gen_seed in gen_seeds:
        print(f"\n{'='*80}")
        print(f"### STARTING PIPELINE FOR GENERATION SEED: {gen_seed} ###")
        print(f"{'='*80}")

        # Middle loop: Model Checkpoint
        for ckpt_path in checkpoint_paths:
            ckpt_name = os.path.basename(ckpt_path)
            print(f"\n{'-'*70}")
            print(f"--- Processing Checkpoint: {ckpt_name} (Gen. Seed: {gen_seed}) ---")
            print(f"{'-'*70}")

            checkpoint_num = int(ckpt_name.replace('model_', '').replace('.pt', ''))
            if checkpoint_num not in args.checkpoints:
                print("Skipping checkpoint: ", ckpt_name)
                continue

            # --- STAGE 1: Generate a SINGLE synthetic dataset ---
            gen_command = [
                sys.executable, '-m', 'sample.synth_dataset_generate',
                '--model_path', ckpt_path,
                '--seed', str(gen_seed)
            ]
            action_desc = f"Generating data for [{ckpt_name}] with seed [{gen_seed}]"
            
            if args.gen_data:
                run_command(gen_command, action_desc)
            else:
                print(f"\n> Skipping data generation for [{ckpt_name}] with seed [{gen_seed}] (set --gen_data if you need to generate the synthetic dataset on-the-fly).")

            # --- Find the specific dataset that was just created ---
            n_iter = ckpt_name.replace('model_', '').replace('.pt', '')
            synth_dir_name = f'{exp_name}_{n_iter}_seed{gen_seed}'
            # Use wildcards for dataset-specific parts of the path
            search_pattern = os.path.join('./dataset', '*', 'splits', 'synth', '*', '*', synth_dir_name)
            found_dirs = glob.glob(search_pattern)

            if not found_dirs:
                print(f"\nError: Could not find the generated dataset directory for '{synth_dir_name}'.")
                print("Skipping HAR training for this dataset.")
                continue  # Skip to the next checkpoint
            
            synth_dir = found_dirs[0]
            print(f"✅ Successfully created and found dataset: {synth_dir}")

            # --- STAGE 2: Train all HAR models for this specific dataset ---
            print(f"\n--- Launching HAR training runs for: {os.path.basename(synth_dir)} ---")
            # Inner loops: Augmentations and Training Seeds
            for use_augs in augmentation_options:
                for train_seed in train_seeds:
                    # Build the command for train.train_har
                    train_command = [
                        sys.executable, '-m', 'train.train_har',
                        '--data_dir', synth_dir,
                        '--seed', str(train_seed),
                        '--mode', args.mode,
                        '--data_rep', args.data_rep,
                        '--model_har', args.model_har,
                        '--pyskl', args.pyskl,
                    ]
                    
                    # Add boolean and optional flags
                    if use_augs:
                        train_command.append('--augs')
                    if args.keep_data:
                        train_command.append('--keep_data')
                    if args.use_wandb:
                        train_command.append('--use_wandb')
                        train_command.extend(['--wandb_project', args.wandb_project])

                    augs_text = "with augmentations" if use_augs else "without augmentations"
                    action_desc = f"Training HAR model ({augs_text}, seed {train_seed})"
                    run_command(train_command, action_desc)

    print("\n🎉🎉🎉 Full evaluation pipeline finished successfully! 🎉🎉🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a full downstream evaluation pipeline: sequentially generate a synthetic dataset "
                    "from a checkpoint, then immediately train all HAR models on it."
    )
    
    # --- Required Arguments ---
    parser.add_argument(
        '--model_dir', 
        type=str, 
        required=True,
        help="Directory containing the model checkpoints (e.g., './save/my_experiment')."
    )

    # --- Arguments for train.train_har ---
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[3000], help='List of checkpoint numbers to evaluate.')
    parser.add_argument('--mode', type=str, default='mixed', choices=['gt', 'real', 'synth', 'mixed'], help='Training mode for the HAR model.')
    parser.add_argument('--data_rep', type=str, default='j', choices=['b', 'bm', 'j', 'jm'], help='Data representation format for the HAR model.')
    parser.add_argument('--model_har', type=str, default='stgcn', choices=['stgcn'], help='HAR model architecture to use.')
    parser.add_argument('--pyskl', type=str, default='../pyskl', help='Path to the local PySKL repository.')
    parser.add_argument('--keep_data', action='store_true', help='Whether to keep the temporary PySKL dataset file after training.')
    parser.add_argument('--gen_data', action='store_true', help='Perform dataset generation before training HAR model.')
    # --- W&B Logging Arguments ---
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging for the HAR training.')
    parser.add_argument('--wandb_project', type=str, default='har_eval', help='Name of the W&B project.')
    
    args = parser.parse_args()
    
    main(args)