
from config  import config
from model import siMLPe as Model

import torch
from torch.utils.data import DataLoader

import test

def test(config, model, dataloader) :

    print("Testing with component size %d"%config.data_component_size)
    
    m_p3d_h36 = np.zeros([config.motion.h36m_zed_target_length])
    titles = np.array(range(config.motion.h36m_zed_target_length)) + 1
    #joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    joint_used_xyz= np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24]).astype(np.int64)
    num_samples = 0

    pbar = dataloader
    #m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36, data_component_size = config.data_component_size)

    ret = {}
    for j in range(config.motion.h36m_zed_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
args = parser.parse_args()

model = Model(config)
state_dict = torch.load(args.model_dict)
model.load_state_dict(state_dict, strict = True)

model.eval()
model.cuda()

dataset = H36MEval(config, 'test')

dataloader = DataLoader(dataset, batch_size = 128, num_workers = 1, drop_last = False,
                        sampler = sampler, shuffle = shuffle, pin_memory = True)

a, b = dataset[10]

traced_script_module = torch.jit.trace(model, example)
