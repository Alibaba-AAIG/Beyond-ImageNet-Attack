import torch
import torchvision
from generator import GeneratorResnet
import pandas as pd


# Load a particular generator
def load_gan(args, domain): 
    if domain[-5:] == 'incv3': 
        netG = GeneratorResnet(inception=True)
    else:
        netG = GeneratorResnet()

    
    if args.RN and args.DA:
        save_checkpoint_suffix = 'BIA+RN+DA'
    elif args.RN:
        save_checkpoint_suffix = 'BIA+RN'
    elif args.DA:
        save_checkpoint_suffix = 'BIA+DA'
    else:
        save_checkpoint_suffix = 'BIA'  

    print('Substitute Model: {} \t RN: {} \t DA: {} \tSaving instance: {}'.format(args.model_type,
                                                                                  args.RN,
                                                                                  args.DA,
                                                                                  args.epochs))
                                                                                                           
    netG.load_state_dict(torch.load('saved_models/{}/netG_{}_{}.pth'.format(args.model_type,
                                                                            save_checkpoint_suffix,
                                                                            args.epochs)))

    return netG
