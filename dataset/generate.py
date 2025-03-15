import os
import pathlib
import random
import argparse

def generate_random_numbers(count, max_value):
    return [f"{random.randint(0,max_value):08d}" for _ in range(count)]

def split_data(data, train_ratio, val_ratio):
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    train_set = data[:train_size]
    val_set = data[train_size:train_size + val_size]
    test_set = data[train_size + val_size:]
    return train_set, val_set, test_set

def write_data_to_file(data, filename):
    with open(filename, "w") as f:
        for item in data:
            f.write(item+"\n")

parser = argparse.ArgumentParser(description='Brep2Seq Generate Dataset')
parser.add_argument('--max', type=int, default=10000, help='max number of dataset')
args = parser.parse_args()

random_numbers = generate_random_numbers(args.max, args.max)
train, val, test = split_data(random_numbers, 0.95, 0.025)
write_data_to_file(train, "train.txt")
write_data_to_file(val, "val.txt")
write_data_to_file(test, "test.txt")
print("Done generating dataset")
