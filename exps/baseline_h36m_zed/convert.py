import argparse
import torch

from model import siMLPe as Model
from datasets.h36m_eval import H36MZedDataset
import test

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
parser.add_argument('--data-item', type=int, default=0, help='Dataset entry')
args = parser.parse_args()

model = Model(config)
state_dict = torch.load(args.model_pth)
model.load_state_dict(state_dict, strict = True)
model.eval()
model.cuda()


config.motion.h36m_zed_target_length = config.motion.h36m_zed_target_length_eval

dataset = H36MZedDataset(config, 'train')
data_input, data_target = dataset[args.data_item]
traced_script_module = torch.jit.trace(model, data_input)



