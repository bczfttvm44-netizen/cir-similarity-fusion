from argparse import ArgumentParser

from torch.cuda.amp import autocast
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import CIRDataset
from utils import collate_fn, save_model, RunningAverage
from collections import OrderedDict
from models import CIRPlus
from torch.optim import lr_scheduler

def main():
    model = CIRPlus(args.dataset, tau=args.tau, device=device)
    param_groups = [{'params': model.net.parameters(), 'lr':args.learning_rate}]
    optimizer = optim.AdamW(
        param_groups
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.05)
    loss_avg = RunningAverage()
    
    relative_train_dataset = CIRDataset(args.dataset, 'train', 'relative',args.data_path,
                                        args.dress_types)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=args.batch_size,
                                       num_workers=4, pin_memory=False, collate_fn=collate_fn, shuffle=True,
                                       drop_last=True)
    
    
    model.extract_bank_features()
    relative_train_dataset.use_bank = True
        
    print('Training loop started')
    for epoch in range(args.num_epochs):
        with tqdm(total=len(relative_train_loader)) as t:
            for idx, batch_data in enumerate(relative_train_loader):
                optimizer.zero_grad()
                captions, indexs, target_indexs, target_index_all, reference_index_all = batch_data
                with autocast():
                    loss_dict = model.forward(captions, indexs, target_index_all,
                                                  reference_index_all)
                loss = loss_dict['bank_loss']
                loss_avg.update(loss.item())
                loss_dict['loss'] = loss_avg()
                for loss_name in loss_dict:
                    loss_dict[loss_name] = '{:05.3f}'.format(loss_dict[loss_name])
                t.set_postfix(OrderedDict(loss_dict))
                t.update()
                loss.backward()
                optimizer.step()
                scheduler.step()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'], 2e-4) 
        print(f"Epoch [{epoch}] Loss: {loss_avg()}")
    save_model(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['fiq', 'cirr'],
                        help="should be either 'cirr' or 'fiq'")
    parser.add_argument("--num-epochs", default=5, type=int, help="number training epochs")
    parser.add_argument("--learning-rate", default=2e-5, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=256, type=int, help="Batch size")
    parser.add_argument("--tau", default=0.02, type=float)
    parser.add_argument("--data_path", default='')
    parser.add_argument("--dress_types", default='dress,shirt,toptee')

    args = parser.parse_args()
    if args.data_path == '':
        if args.dataset == 'fiq':
            args.data_path = 'fashionIQ_dataset'
        else:
            args.data_path = 'cirr_dataset'
    device = torch.device('cpu')
    args.dress_types = args.dress_types.split(',')
    print('Arguments:')
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))
    main()
