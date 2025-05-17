import os
import argparse
import time
import subprocess
import concurrent.futures
import numpy as np
import re
import sys
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

window_size = 16
device = 'cuda:0'

for config in ['partial_sorting_4_config_cotrain']:

    seeds = [10,20,30,40,50,60]
    data_ratios = [5, 10, 25]

    eval_records = []
    for data_ratio in data_ratios:
        # training, specify max_workers for multiprocess

        commands1 = []
        for seed in seeds:
            ## run baseline action policy learning
            agent_name = f'df_policy_{data_ratio}_{seed}'
            command1 = (f'python run.py --config-name={config} agents=df_policy agent_name={agent_name} '
                        f'action_data_ratio={data_ratio} window_size={window_size} seed={seed} device={device} '
                        f'trainset.only_label_data=True trainset.obs_data_ratio={data_ratio}')

            # os.system(command1)
            commands1.append(command1)
        # ##Running commands in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            executor.map(run_command, commands1)

        # Eval with multiple cores.
        test_result = []
        for seed in seeds:
            agent_name = f'df_policy_{data_ratio}_{seed}'
            # evaluate baseline policy
            command2 = (
                f'python run_sim_baseline.py --config-name={config} agents=df_agent agent_name={agent_name} '
                f'action_data_ratio={data_ratio} window_size={window_size} seed={seed} device={device} '
                f'trainset.only_label_data=True  trainset.obs_data_ratio={data_ratio} '
                f'simulation.render=False simulation.n_cores=4 ')
            # os.system(command2)

            # summarize results
            print(f"running eval for action_data_ratio:{data_ratio},seed:{seed}")
            result = subprocess.run(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Check for errors
            if result.returncode != 0:
                print(f"Error in command: {command2}")
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
        record = f"action_data_ratio:{data_ratio},results: {test_result_np}, mean: {test_result_np.mean(axis=0)}, std: {test_result_np.std(axis=0)}"
        print(record)

        eval_records.append(record)

    print(eval_records)