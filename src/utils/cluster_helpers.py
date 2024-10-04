import os
import subprocess
import time

import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def submit_to_condor(env_id, exec_cmd, results_dir, job_name, expt_params, user_email="dummy@gmail.com",
                     num_trials=1, sleep_time=0, print_without_submit=False):
    '''purpose of this function is to submit a script to condor that runs num_trials instances
    '''    
    if num_trials == 0: 
        print(f"0 jobs submitted to condor for {results_dir + job_name}, {env_id}")
        return 

    condor_log_dir = os.path.join(results_dir, 'condor_logs')
    if not os.path.exists(condor_log_dir):
        os.makedirs(condor_log_dir)
    
    notification = "Never" # ["Complete", "Never", "Always", "Error"]
    # note that as of 2024, the only GPU clusters are Eldar and Nandor
    # we explicitly specify eldar because jobs fail on nandor
    # previously, the requirements line was:
    # Requirements = (TARGET.GPUSlot) && InMastodon
    condor_contents = \
f"""executable = {exec_cmd} 
universe = vanilla
getenv = true
+GPUJob = true
Requirements = (TARGET.GPUSlot) && Eldar && InMastodon

+Group = "GRAD" 
+Project = "AI_ROBOTICS"
+ProjectDescription = "{job_name} {env_id}"

input = /dev/null
error = {condor_log_dir}/{job_name}_$(CLUSTER).err
output = {condor_log_dir}/{job_name}_$(CLUSTER).out
log = {condor_log_dir}/{job_name}_$(CLUSTER).log

notify_user = {user_email}
notification = {notification}

arguments = \
""" 
    for k, v in expt_params.items():
        condor_contents += f" --{k} {v}" 
    condor_contents += f"\nQueue {num_trials}"

    if print_without_submit: 
        print("CONDOR SUB SCRIPT IS \n", condor_contents)
    else:
        # submit to condor
        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        proc.stdin.write(condor_contents.encode())
        proc.stdin.close()

    time.sleep(sleep_time)
    print(f"Submitted {num_trials} jobs for {job_name}, {env_id} to condor")


def check_tb_logs(savefile_dir, expected_rew_len):
    f = glob.glob(savefile_dir + "/events*")[0]
    iterator = EventAccumulator(f).Reload()
    tag = "test/reward"
    events = iterator.Scalars(tag)
    rewards = [e.value for e in events]
    if len(rewards) < expected_rew_len:
        return False
    return True

def check_single_dir(savefile_dir, check_tb_log=False, expected_rew_len=None):
    if not os.path.exists(savefile_dir):  # directory doesn't even exist
        return False
    if not os.listdir(savefile_dir):  # directory exists but is empty
        return False
    if check_tb_log:
        return check_tb_logs(savefile_dir, expected_rew_len=expected_rew_len)

    return True

def count_nonempty_dirs(savefile_dirlist):
    '''
    Checks number of nonempty and/or existing directories in the list of directories
    '''
    done = []
    for savefile_dir in savefile_dirlist:
        done.append(check_single_dir(savefile_dir))
    return sum(done)