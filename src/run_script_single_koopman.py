import os
import argparse
import time
import concurrent.futures
import subprocess
import numpy as np
import re
import sys

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in process.stdout:
        if '\r' not in line:
            sys.stdout.write(line)
            sys.stdout.flush()
    for line in process.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()
    process.wait()  # Wait for the process to complete
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
    

parser = argparse.ArgumentParser()
parser.add_argument("--data_ratio", type=int, default=0.5, help="a")
parser.add_argument("--net", type=str, default=None, help="a")
opt = parser.parse_args()


time.sleep(0)

window_size = 16
method = 'cotrain_single_koopman'
device = 'cuda:0'

for config in ['partial_sorting_4_config_cotrain']:

    seeds = [40,50,60]
    act_data_ratios = [25]
    obs_data_ratio = 25
    only_label_data = False
    action_loss_factor = 1.0

    backbone='lstm'
    hidden_dims = '[256]'
    latent_dim = 512
    target_k=8

    eval_records = []
    for action_data_ratio in act_data_ratios:
        commands1 = []

        for seed in seeds:
            idm_model_name = f'ablation2_idm_{method}_latent{latent_dim}_data{action_data_ratio}_seed{seed}'
            command1 = (f'python run.py --config-name={config} '
                        f'agents=idm_{method} agent_name={idm_model_name} window_size={window_size} method={method} seed={seed} '
                        f'kpm_latent_dim={latent_dim} target_k={target_k} action_data_ratio={action_data_ratio} action_loss_factor={action_loss_factor} '
                        f'backbone={backbone} trainset.only_label_data={only_label_data} hidden_dims={hidden_dims} trainset.obs_data_ratio={obs_data_ratio} ' #trainset.obs_data_ratio={obs_data_ratio}
                        f'device={device}')
            # os.system(command1)
            commands1.append(command1)

        ## Running commands in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            executor.map(run_command, commands1)

        test_result = []
        for seed in seeds:
            idm_model_name = f'ablation2_idm_{method}_latent{latent_dim}_data{action_data_ratio}_seed{seed}'
            command3 = (
                f'python run_sim.py --config-name={config} '
                f'agents=idm_agent_{method} agent_name={idm_model_name} window_size={window_size} method={method} seed={seed} '
                f'action_data_ratio={action_data_ratio} action_loss_factor={action_loss_factor} '
                f'kpm_latent_dim={latent_dim} target_k={target_k} backbone={backbone} trainset.only_label_data={only_label_data} hidden_dims={hidden_dims} trainset.obs_data_ratio={obs_data_ratio} '#trainset.obs_data_ratio={obs_data_ratio}
                f'simulation.render=True device={device} simulation.n_cores=1')
            # os.system(command3)

            print(f"running eval for action_data_ratio:{action_data_ratio},seed:{seed}")
            result = subprocess.run(command3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Check for errors
            if result.returncode != 0:
                print(f"Error in command: {command3}")
                print("Errors:", result.stderr)
                continue

            # Process the output
            try:
                match = re.search(r'eval success rate:(\d+\.\d+)', result.stdout)
                if match:
                    eval_success_rate = float(match.group(1))

                print(f"result for seed:{seed}, success_rate:{eval_success_rate}")
                test_result.append(eval_success_rate)
            except ValueError as e:
                print(f"Error processing output from seed {seed}: {result.stdout}")
                print(e)
                continue

            # Convert the list to a numpy array
        test_result_np = np.asarray(test_result)

        # Print the results, mean, and standard deviation
        record = f"action_data_ratio:{action_data_ratio},results: {test_result_np}, mean: {test_result_np.mean()}, std: {np.std(test_result_np)}"
        print(record)

        eval_records.append(record)

    print(eval_records)

