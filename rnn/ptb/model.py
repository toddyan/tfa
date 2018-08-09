import numpy as np
import globalconf

def file_to_id_list(id_file):
    with open(id_file) as f:
        return list(map(
            lambda x:int(x),
            ' '.join([line.strip() for line in f.readlines()]).split(' ')
        ))


def make_batch(id_list, batch_size, timestep):
    num_batchs = (len(id_list)-1) // (batch_size * timestep)
    x = np.array(id_list[:num_batchs * batch_size * timestep])
    x = np.reshape(x, [batch_size, num_batchs * timestep])
    x = np.split(x, num_batchs, axis=1) # [num_batchs, batch_size, timestep]
    y = np.array(id_list[1:num_batchs * batch_size * timestep+1])
    y = np.reshape(y, [batch_size, num_batchs * timestep])
    y = np.split(y, num_batchs, axis=1) # [num_batchs, batch_size, timestep]
    return list(zip(x,y))


if __name__ == "__main__":
    train_file = globalconf.get_root() + "rnn/ptb/simple-examples/data/id.train.txt"
    r = file_to_id_list(train_file)
    print(r[0:1000])
    batch_size = 20
    timestep = 35
    data = make_batch(r, batch_size, timestep)
    for e in data:
        print(np.array(e).shape) #(2, 20, 35)
        exit()