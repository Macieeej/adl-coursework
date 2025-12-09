import sys
import re
import csv

hyperparam_pattern = re.compile(r"weight decay: ([0-9.e-]+), learning rate: ([0-9.e-]+), dropout: ([0-9.e-]+)")
mean_auc_pattern = re.compile(r"Best checkpoint: (\d+) with mean auc score of ([0-9.]+)")
mean_auc_test_pattern = re.compile(r"Mean AUC score for test set: ([0-9.]+)")

def main(args):
    if len(args) < 2:
        print("Usage: %s <JobID>" % args[0])
        return 1
    jobid = args[1]

    common_name = "output" + jobid + "_"
    #print("Common name is %s" % common_name)

    # problem #27: Python does not let you declare variables and not initailize them
    # Hyperparameters
    weight_decay = None
    learning_rate = None
    dropout = None

    # Mean AUC for given checkpoint
    mean_auc = None
    checkpoint = None

    # Mean AUC for test set
    mean_auc_test = None

    # List of collected entries
    entries = []

    for i in range(0, 125):
        dir_name = common_name + str(i)
        #print("Checking directory %s" % dir_name)
        with open(dir_name + "/log_" + jobid + "_" + str(i) + ".out") as f:
            lines = f.readlines()
            for line in lines:
                match = hyperparam_pattern.search(line)
                if match:
                    weight_decay = match.group(1)
                    learning_rate = match.group(2)
                    dropout = match.group(3)
                match = mean_auc_pattern.search(line)
                if match:
                    checkpoint = match.group(1)
                    mean_auc = match.group(2)
                match = mean_auc_test_pattern.search(line)
                if match:
                    mean_auc_test = match.group(1)
            entries.append((weight_decay, learning_rate, dropout, checkpoint, mean_auc, mean_auc_test))
            print(f"Found entry: wd: {weight_decay}, lr: {learning_rate}, dropout: {dropout}, \n" + " " * 13 + f"checkpoint: {checkpoint}, mean_auc: {mean_auc}, test_auc: {mean_auc_test}")
    with open("results_" + jobid + ".csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["weight_decay", "learning_rate", "dropout", "checkpoint", "mean_auc", "mean_auc_test"])
        writer.writeheader()
        for entry in entries:
            writer.writerow({"weight_decay": entry[0], "learning_rate": entry[1], "dropout": entry[2], "checkpoint": entry[3], "mean_auc": entry[4], "mean_auc_test": entry[5]})

if __name__ == "__main__":
    main(sys.argv)