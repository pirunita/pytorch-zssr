import os
import tqdm

import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from networks import ZSSRNetwork
from options import get_opt
from torchvision import transforms
from utils import DataHandler, adjust_learning_rate





def train(opt, model, input_img):
    model.cuda()
    
    handler = DataHandler(opt, input_img)
    
    learning_rate = opt.learning_rate
    min_learning_rate = opt.min_learning_rate
    learning_rate_change_iter_nums = [0]
    mse_steps = []
    mse_rec = []
    
    criterionL1 = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    
    with tqdm.tqdm(miniters=1, mininterval=0) as progress:
        for iter, (hr, lr) in enumerate(handler.preprocess_data()):
            lr = lr.cuda()
            hr = hr.cuda()
            
            output = model(lr) + lr
            loss = criterionL1(output, hr)
            
            cpu_loss = loss.data.cpu().numpy()
            
            
            
            model.zero_grad()
            optimizer.zero_grad()
            
            progress.set_description("Iteration: {} Loss: {}, Learning rate: {}".format( \
                iter, cpu_loss, learning_rate))
            progress.update()
            
            if iter > 0 and iter % 10000 == 0:
                learning_rate = learning_rate / 10
                adjust_learning_rate(optimizer, new_lr=learning_rate)
                print("Learning rate reduced to {lr}".format(lr=learning_rate) )
            """
            if (not (1 + iter) % opt.learning_rate_policy_check_every
                and iter - learning_rate_change_iter_nums[-1] > opt.min_iters):
                [slope, _], [[var, _], _] = np.polyfit(mse_steps[-int(opt.learning_rate_slope_range /
                                                                    opt.run_test_every):],
                                                        mse_rec[-int(opt.learning_rate_slope_range /
                                                                    opt.run_test_every):],
                                                        1, cov=True)
                
                std = np.sqrt(var)
                
                if -opt.learning_rate_change_ratio * slope < std:
                    learning_rate /= 10
                    learning_rate_change_iter_nums.append(iter)
            """
                
            loss.backward()
            optimizer.step()
            
            if learning_rate < min_learning_rate:
                print('Done training')
                break
            
            
def test(opt, model, input_img):
    model.eval()
    
    input_img = input_img.resize((int(input_img.size[0]*opt.scale_factor), \
                                int(input_img.size[1]*opt.scale_factor)), resample=Image.BICUBIC)
    input_img.save('low_res.png')
    
    input_img = transforms.ToTensor()(input_img)
    input_img = input_img.unsqueeze(0)
    input = torch.autograd.Variable(input_img.cuda())
    
    output = input + model(input)
    output = output.cpu().data[0, :, :, :]
    
    output_na = output.numpy()
    
    # Normalization
    output_na[np.where(output_na < 0)] = 0.0
    output_na[np.where(output_na > 1)] = 1.0
    
    output_img = torch.from_numpy(output_na)
    output_img = transforms.ToPILImage()(output_img)
    output_img.save('zssr.png')
    
    

def main():
    opt = get_opt()
    
    input_img = Image.open(opt.input_img)
    model = ZSSRNetwork()
    
    train(opt, model, input_img)
    test(opt, model, input_img)
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()