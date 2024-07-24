from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import csv
import logging
import abc
import tensorflow as tf
import numpy as np
import src.resources as resources
import constants
from queue import PriorityQueue
import src.vm_job as defs
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import os
from datetime import datetime

tf.compat.v1.enable_v2_behavior()

logging.basicConfig(level=logging.INFO, filename=constants.root+'/output/'+constants.algo+'.log', filemode='w')

episodes = 1
# Create a new folder named with the timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_env")
output_folder = os.path.join(constants.root, 'output', timestamp)
os.makedirs(output_folder, exist_ok=True)



class ClusterEnv(py_environment.PyEnvironment):

    def __init__(self):
        # Adjusting the code to save the results CSV file in the new directory
        self.file_result = open(os.path.join(output_folder, 'results_' + constants.algo + '.csv'), 'a+', newline='')
        self.episode_reward_writer = csv.writer(self.file_result, delimiter=',')
        self.episode_reward_writer.writerow(["Episode", "Reward", "Cost", "AVGtime", "GoodPlacement", "ResourceUtilization", "Throughput", "DeadlineAdherence"])
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=9, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(resources.features,), dtype=np.int32, minimum=resources.cluster_state_min,
            maximum=resources.cluster_state_max,
            name='observation')
        self._state = copy.deepcopy(resources.cluster_state_init)
        self._episode_ended = False
        self.reward = 0
        self.vms = copy.deepcopy(resources.VMS)
        self.jobs = copy.deepcopy(resources.JOBS)
        self.clock = self.jobs[0].arrival_time
        self.job_idx = 0
        self.job_queue = PriorityQueue()
        self.episode_success = False
        self.good_placement = 0
        self.total_cpu_used = 0
        self.total_mem_used = 0
        self.completed_jobs = 0
        self.cpu_utilization_history = []
        self.mem_utilization_history = []

        # Initialize total_cpu and total_mem
        self.total_cpu = sum([vm.cpu for vm in self.vms])
        self.total_mem = sum([vm.mem for vm in self.vms])

        


    def get_resource_utilization(self):
        return self.cpu_utilization_history, self.mem_utilization_history


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def get_step_reward(self):
        return self.reward

    def _reset(self):
        # cluster.init_cluster()
        self._state = copy.deepcopy(resources.cluster_state_init)
        self._episode_ended = False
        self.reward = 0
        self.vms = copy.deepcopy(resources.VMS)
        self.jobs = copy.deepcopy(resources.JOBS)
        self.clock = self.jobs[0].arrival_time
        self.job_idx = 0
        self.job_queue = PriorityQueue()
        self.episode_success = False
        self.good_placement = 0
        self.total_cpu_used = 0
        self.total_mem_used = 0
        self.completed_jobs = 0
        self.cpu_utilization_history = []
        self.mem_utilization_history = []
        self.start_time = self.clock  # Capture the start time at the beginning of the episode
        # print(self.jobs[self.job_idx].ex_placed)
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):

        global episodes
        # logging.debug("Current Cluster State: {}".format(self._state))
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            # print(' i was here to reset and episode!!!!!!!!!!!!!!1')
            return self.reset()

        if action > 9 or action < 0:
            raise ValueError('`action` should be in 0 to 9.')

        elif action == 0:
            logging.debug("CLOCK: {}: Action: {}".format(self.clock, action))
            # penalty for partial placement
            if self.jobs[self.job_idx].ex_placed > 0:
                self.reward = (-50)
                self._episode_ended = True
                logging.info("CLOCK: {}: Partial Executor Placement for a Job. Episode Ended\n\n".format(self.clock))
            # if no running jobs but jobs waiting to be scheduled -> huge Neg Reward and episode ends
            elif self.job_queue.empty():
                self.reward = (-200)
                self._episode_ended = True
                logging.info(
                    "CLOCK: {}: No Executor Placement When No Job was Running. Episode Ended\n\n".format(self.clock))
            # finishOneJob() <- finish one running job, update cluster states-> "self._state"
            else:
                self.reward = -1
                _, y = self.job_queue.get()
                self.clock = y.finish_time
                self.finish_one_job(y)
            # TODO add check for large job which does not fit in the cluster
        else:
            logging.info("CLOCK: {}: Action: {}".format(self.clock, action))
            # if valid placement, place 1 ex in the VM chosen, update cluster states -> "self._state";
            # check for episode end  -> update self._episode_ended
            if self.execute_placement(action):
                # print('placement successful, clock: ', self.clock)
                if self.check_enough_cluster_resource():
                    self.reward = 1
                else:
                    self.reward = (-200)
                    self._episode_ended = True
                    logging.info(
                        "CLOCK: {}: Optimistic Executor Placement will lead to cluster resource shortage. Episode "
                        "Ended\n\n".format(self.clock))
                # TODO Episode end check needed or not?
                # self.check_episode_end()
            # if invalid placement -> Huge Neg Reward and episode ends
            else:
                self.reward = (-200)
                self._episode_ended = True
                logging.info("CLOCK: {}: Invalid Executor Placement, Episode Ended\n\n".format(self.clock))

            # self._episode_ended = True -> when last job's last executor is placed or bad action

            # self._state = generate new state after executing the current action
        if self._episode_ended:

            end_time = self.clock  # Capture the end time at the end of the episode
            self.total_time = end_time - self.start_time

            epi_cost = resources.max_episode_cost
            epi_avg_job_duration = resources.min_avg_job_duration + \
                                   resources.min_avg_job_duration * float(constants.placement_penalty) / 100
            #epi_cost = self.calculate_vm_cost()
            #epi_avg_job_duration = self.calculate_avg_time()
            resource_utilization = self.calculate_average_utilization()
            throughput = self.calculate_throughput()  # Ensure throughput is calculated here
            deadline_adherence = self.calculate_deadline_adherence()


            if self.episode_success:
                # Multi-Objective Reward Calculation
                epi_cost = self.calculate_vm_cost()
                cost_normalized = 1 - (epi_cost / resources.max_episode_cost)
                #cost_reward = cost_normalized * constants.beta
                cost_reward = cost_normalized * 0.0
                #print(f"Cost: {epi_cost}, Cost Reward: {cost_reward}")

                epi_avg_job_duration = self.calculate_avg_time()
                max_avg_job_duration = resources.min_avg_job_duration + resources.min_avg_job_duration * (constants.placement_penalty/100.0)
                time_normalized = 1 - (epi_avg_job_duration-resources.min_avg_job_duration) / (max_avg_job_duration-resources.min_avg_job_duration)
                #time_reward = time_normalized * (1 - constants.beta)
                time_reward = time_normalized * 0.0
                #print(f"Average Job Duration: {epi_avg_job_duration}, Time Reward: {time_reward}")

                # resource_utilization = self.calculate_average_utilization()
                # utilization_reward = resource_utilization * constants.beta

                # Normalize the resource utilization
                resource_utilization = self.calculate_average_utilization()  # Already normalized between 0 and 1
                utilization_normalized = 1 - resource_utilization
                utilization_reward = utilization_normalized * 0.0
                #print(f"Resource Utilization: {resource_utilization}, Utilization Reward: {utilization_reward}")


                # Calculate throughput reward
                throughput = self.calculate_throughput()
                max_throughput = len(self.jobs) / self.jobs[-1].arrival_time if self.jobs else 1
                throughput_normalized = throughput / max_throughput
                throughput_reward = throughput_normalized * 1.0  # Adjust the weight as needed

                #calculate the Deadline Adherance reward
                adherence_reward = deadline_adherence * 0.0  # Adjust the weight as needed




                self.reward = constants.fixed_episodic_reward * (
                    cost_reward + time_reward + utilization_reward + throughput_reward + adherence_reward)

                #print(f"Total Reward: {self.reward}")

                # Log and write results
                logging.info("CLOCK: {}: ****** Episode ended Successfully!!!!!!!! \n\n".format(self.clock))
                logging.info("cost normalized: {}, cost reward: {}, time normalized: {}, time reward: {}, utilization reward: {}, throughput reward: {}, adherence reward: {}, final reward: {}\n\n".format(cost_normalized, cost_reward, time_normalized, time_reward, utilization_reward, throughput_reward, adherence_reward, self.reward))

            # Write results for an episode
            self.episode_reward_writer.writerow([episodes, self.reward, epi_cost, epi_avg_job_duration, self.good_placement, resource_utilization, throughput])
            episodes += 1
            return ts.termination(np.array(self._state, dtype=np.int32), self.reward)

        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=self.reward, discount=.9)
        

    

    def calculate_throughput(self):
        total_time = self.clock  # Assuming `self.clock` represents the total time passed
        if total_time > 0:
            throughput = self.completed_jobs / total_time
        else:
            throughput = 0
        return throughput



    def finish_one_job(self, finished_job):
        finished_job.finished = True
        finished_job.running = False
        for i in range(len(finished_job.ex_placement_list)):
            vm = self.vms[finished_job.ex_placement_list[i]]
            vm.cpu_now += finished_job.cpu
            vm.mem_now += finished_job.mem
            self.completed_jobs += 1
            self.total_cpu_used += finished_job.cpu * finished_job.duration
            self.total_mem_used += finished_job.mem * finished_job.duration
            # TODO copy/reference needed or not?
            # self.vms[vm.id] = vm
        self._state = resources.gen_cluster_state(self.job_idx, self.jobs, self.vms)
        self.update_resource_utilization()  # Update utilization after job completion
        logging.info("CLOCK: {}: Finished execution of job: {}".format(self.clock, finished_job.id))
        logging.debug("CLOCK: {}: Current Cluster State: {}".format(self.clock, self._state))



    def execute_placement(self, action):
        # retrives the current job and determine the VM index
        current_job = self.jobs[self.job_idx]
        vm_idx = action - 1
        # check if selected VM has enough resources
        if current_job.cpu > self.vms[vm_idx].cpu_now or current_job.mem > self.vms[vm_idx].mem_now:
            return False

        #update the VMs resource allocation
        self.vms[vm_idx].cpu_now -= current_job.cpu
        self.vms[vm_idx].mem_now -= current_job.mem

        self.total_cpu_used += current_job.cpu
        self.total_mem_used += current_job.mem

        # update the job placement information
        current_job.ex_placed += 1
        current_job.ex_placement_list.append(vm_idx)
        # print('current job variable executor: {}\n'.format(current_job.ex_placed))
        # print('self job variable executor: {}\n'.format(self.jobs[self.job_idx].ex_placed))
        # TODO deep copy needed or not?
        # self.jobs[self.job_idx] = copy.deepcopy(current_job)

        # check if the job is fully placed
        if current_job.ex_placed == current_job.ex:
            # self.reward = 10
            # log the completion of the job placement
            logging.info("CLOCK: {}: Finished placement of job: {}".format(self.clock, current_job.id))
            # Apply Job Duration Variance depending on placement type
            # For CPU and Memory bound applications -> Consolidated placement is better
            # For IO / Network bound applications -> Distributed placement is better
            # If condition does not satisfy -> Apply a 20% job duration increase


            # apply job duration variance based on placement type
            if constants.pp_apply == 'true':
                # IO / Network bound jobs
                if current_job.type == 3:
                    if len(set(current_job.ex_placement_list)) != 1:
                        logging.debug("***** Bad placement for type 3 job. Executors: {}, Machines used: {}".format(
                            current_job.ex_placed, len(set(current_job.ex_placement_list))))
                        duration_increase = current_job.duration * float(constants.placement_penalty) / 100
                        current_job.duration += duration_increase
                    else:
                        self.good_placement += 1
                        logging.debug("***** Good placement for type 3 job. Executors: {}, Machines used: {}".format(
                            current_job.ex_placed, len(set(current_job.ex_placement_list))))
                # Compute or Memory bound jobs
                else:
                    if len(set(current_job.ex_placement_list)) < current_job.ex_placed:
                        logging.debug("***** Bad placement for type 1 or 2 job. Executors: {}, Machines used: {}".format
                                      (current_job.ex_placed, len(set(current_job.ex_placement_list))))
                        duration_increase = current_job.duration * float(constants.placement_penalty) / 100
                        current_job.duration += duration_increase
                    else:
                        self.good_placement += 1
                        logging.debug(
                            "***** Good placement for type 1 or 2 job. Executors: {}, Machines used: {}".format(
                                current_job.ex_placed, len(set(current_job.ex_placement_list))))
            # Update current job start and finish times
            current_job.running = True
            current_job.start_time = self.clock
            current_job.finish_time = self.clock + current_job.duration
            # Update VM usage data for each VM used for placing executors of the current job
            for i in range(len(current_job.ex_placement_list)):
                if current_job.start_time > self.vms[current_job.ex_placement_list[i]].stop_use_clock:
                    self.vms[current_job.ex_placement_list[i]].used_time += current_job.duration
                    self.vms[current_job.ex_placement_list[i]].stop_use_clock = current_job.finish_time
                else:
                    if current_job.finish_time > self.vms[current_job.ex_placement_list[i]].stop_use_clock:
                        self.vms[current_job.ex_placement_list[i]].used_time += (
                                current_job.finish_time - self.vms[current_job.ex_placement_list[i]].stop_use_clock)
                        self.vms[current_job.ex_placement_list[i]].stop_use_clock = current_job.finish_time
            # TODO deep copy needed or not?
            # self.jobs[self.job_idx] = copy.deepcopy(current_job)

            # update the job queue
            self.job_queue.put((current_job.finish_time, current_job))
            if self.job_idx + 1 == len(self.jobs):
                self._episode_ended = True
                self.episode_success = True
                self.update_resource_utilization()  # Update utilization after job completion
                return True
            self.job_idx += 1
            self.clock = self.jobs[self.job_idx].arrival_time

            # handle jobs that finish before the next job's arrival
            while True:
                if self.job_queue.empty():
                    break
                _, next_finished_job = self.job_queue.get()
                if next_finished_job.finish_time <= self.clock:
                    self.finish_one_job(next_finished_job)
                else:
                    self.job_queue.put((next_finished_job.finish_time, next_finished_job))
                    break
        
        # generate new cluster state
        self._state = resources.gen_cluster_state(self.job_idx, self.jobs,
                                                self.vms)
        logging.debug("CLOCK: {}: Current Cluster State: {}".format(self.clock, self._state))

        # Track resource utilization
        self.update_resource_utilization()
        return True

    def check_enough_cluster_resource(self):
        # retrieve current job and initialize variables
        current_job = self.jobs[self.job_idx]
        possible_placement = 0
        remaining_placement = current_job.ex - current_job.ex_placed

        # calculate possible job placement across all VMs
        for i in range(len(self.vms)):
            possible_placement += min(self.vms[i].cpu_now / current_job.cpu, self.vms[i].mem_now / current_job.mem)

        return possible_placement >= remaining_placement

    def check_episode_end(self):
        current_job = self.jobs[self.job_idx]
        if self.job_idx + 1 == len(self.jobs) and current_job.ex == current_job.ex_placed:
            self._episode_ended = True

    def calculate_vm_cost(self):
        cost = 0
        for i in range(len(self.vms)):
            cost += (self.vms[i].price * self.vms[i].used_time)
            logging.info("VM: {}, Price: {}, Time: {}".format(i, self.vms[i].price, self.vms[i].used_time))
        logging.info("***Episode VM Cost: {}".format(cost))
        self.episode_cost = cost
        return cost

    def calculate_avg_time(self):
        time = 0
        for i in range(len(self.jobs)):
            time += self.jobs[i].duration
            logging.debug("Job: {}, Duration: {}".format(self.jobs[i].id, self.jobs[i].duration))
        avg_time = float(time) / len(self.jobs)
        logging.info("***Episode AVG Job Duration: {}".format(avg_time))
        return avg_time
    
    def get_vm_cost(self):
        return self.calculate_vm_cost()
    
    def get_avg_time(self):
        return self.calculate_avg_time()

    

    def update_resource_utilization(self):
        total_cpu_used_now = sum([vm.cpu - vm.cpu_now for vm in self.vms])
        total_mem_used_now = sum([vm.mem - vm.mem_now for vm in self.vms])
        cpu_utilization = total_cpu_used_now / self.total_cpu
        mem_utilization = total_mem_used_now / self.total_mem
        self.cpu_utilization_history.append(cpu_utilization)
        self.mem_utilization_history.append(mem_utilization)
        # Print the current history of CPU and memory utilizations
        #print(f"CPU Utilization History: {self.cpu_utilization_history}")
        #print(f"Memory Utilization History: {self.mem_utilization_history}")


    
    def calculate_average_utilization(self):
        avg_cpu_utilization = np.mean(self.cpu_utilization_history) if self.cpu_utilization_history else 0
        avg_mem_utilization = np.mean(self.mem_utilization_history) if self.mem_utilization_history else 0
        avg_utilization = (avg_cpu_utilization + avg_mem_utilization) / 2
        return avg_utilization
    


    def calculate_deadline_adherence(self):
        on_time_jobs = 0
        for job in self.jobs:
            if job.finish_time and job.finish_time <= job.deadline:
                on_time_jobs += 1
        adherence_rate = on_time_jobs / len(self.jobs) if self.jobs else 0
        return adherence_rate
    

    def get_deadline_adherence(self):
        return self.calculate_deadline_adherence()

# environment = ClusterEnv()
# environment2 = ClusterEnv()

# environment = ClusterEnv()
# utils.validate_py_environment(environment, episodes=1000)
