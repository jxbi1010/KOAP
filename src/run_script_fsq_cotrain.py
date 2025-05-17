import os
import argparse
import time
import concurrent.futures
import subprocess
import numpy as np
import re
import sys
# def run_command(command):
#     process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     for line in process.stdout:
#         print(line, end='')  # Print stdout line by line
#     for line in process.stderr:
#         print(line, end='')  # Print stderr line by line
#     process.wait()  # Wait for the process to complete
#     if process.returncode != 0:
#         print(f"Command failed with return code {process.returncode}")


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in process.stdout:
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
method = '_fsq'
device = 'cuda:0'
only_label_data = False


#TODO: tune fsq_latent_dim
for config in ['partial_sorting_4_config_cotrain']:

    seeds = [10,20,30,40,50,60]
    data_ratios = [1,2,5,10,25]
    action_loss_factor = 1.0

    levels = '[8,5,5,5]'
    fsq_latent_dim = 16
    backbone = 'lstm'
    hidden_dims = '[256]'

    eval_records = []
    for action_data_ratio in data_ratios:
        commands1 = []
        for seed in seeds:
            idm_model_name = f'idm_cotrain{method}_latent{fsq_latent_dim}_backbone{backbone}_data{action_data_ratio}_seed{seed}'
            command1 = (f'python run.py --config-name={config} '
                        f'agents=idm_cotrain{method} agent_name={idm_model_name} window_size={window_size} method={method} seed={seed} '
                        f'action_data_ratio={action_data_ratio} action_loss_factor={action_loss_factor} '
                        f'fsq_latent_dim={fsq_latent_dim} hidden_dims={hidden_dims} fsq_levels={levels} backbone={backbone} trainset.only_label_data={only_label_data} '
                        f'device={device}')
            # os.system(command1)
            commands1.append(command1)

        # Running commands in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            executor.map(run_command, commands1)

        test_result = []
        for seed in seeds:
            idm_model_name = f'idm_cotrain{method}_latent{fsq_latent_dim}_backbone{backbone}_data{action_data_ratio}_seed{seed}'
            command3 = (
                f'python run_sim.py --config-name={config} '
                f'agents=idm_agent_cotrain{method} agent_name={idm_model_name} window_size={window_size} method={method} seed={seed} '
                f'action_data_ratio={action_data_ratio} action_loss_factor={action_loss_factor} '
                f'fsq_latent_dim={fsq_latent_dim} hidden_dims={hidden_dims} fsq_levels={levels} backbone={backbone} trainset.only_label_data={only_label_data} '
                f'simulation.render=False device={device} simulation.n_cores=4')
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
