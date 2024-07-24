import csv
import src.vm_job as defs
import constants
JOBS_WORKLOAD = []


def read_workload():
    with open('D:\\sumit\\RM_DeepRL-master\\input\\'+constants.workload) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        global JOBS_WORKLOAD
        JOBS_WORKLOAD = []
        for row in readCSV:
            # a_t, id, type, cpu, mem, ex, duration
            JOBS_WORKLOAD.append(defs.JOB(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6])))
        # print(JOBS_WORKLOAD[0].arrival_time)
        # print(JOBS_WORKLOAD[0].id)
        # print(JOBS_WORKLOAD[0].type)
        # print(JOBS_WORKLOAD[0].cpu)
        # print(JOBS_WORKLOAD[0].mem)
        # print(JOBS_WORKLOAD[0].ex)
        # print(JOBS_WORKLOAD[0].duration)
