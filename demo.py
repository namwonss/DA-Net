import os,argparse
import numpy as np
from DA_Net import DA_Net_t
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
import torch.nn.functional as F
import torch.utils.data as data
from torchinfo import summary
from metrics import ciede2000
from ptflops import get_model_complexity_info
from PIL import Image

parser = argparse.ArgumentParser()
#parser.add_argument('--trained_dir', type=str, default='./trained/DA_Net_RSID_146_3007_0974_2239.pk')
parser.add_argument('--trained_dir', type=str, default='./trained_models/DA-Net_RSID_146.pk')
parser.add_argument('--test_dir', type=str, default='./dataset/RSID/test/')
parser.add_argument('--hazy_dir', type=str, default='hazy')
parser.add_argument('--GT_dir', type=str, default='GT')
parser.add_argument('--gpu', default='0', type=str, help='GPUs for demo')
opt = parser.parse_args()

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class test_Dataset(data.Dataset):
    def __init__(self,path,size=None,format='.png',hazy=opt.hazy_dir,GT=opt.GT_dir):
        super(test_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,hazy))
        self.haze_imgs=[os.path.join(path,hazy,img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,GT)
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split('_')[0]
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear,clear_name
    def augData(self,data,target):
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)


def test(test_loader, network, result_dir=None):

    if not os.path.exists('./output'):
        os.makedirs('./output')
    
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    CIEDE = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0
    for i, (input_imgs, targets, img_name) in enumerate(test_loader):
        inputs = input_imgs.to('cuda')
        target = targets.to('cuda') 
        img_name = img_name[0]
        with torch.no_grad():
            start.record()
            output = network(inputs)
            end.record()
            torch.cuda.synchronize()
            intermediate_time = start.elapsed_time(end)/1000
            total_time += intermediate_time

            ###
            # Save output image
            dehazed_img = torch.squeeze(output.clamp(0, 1).cpu()).permute(1,2,0).numpy()
            dehazed_img = np.round(dehazed_img * 255)
            img_array = Image.fromarray((dehazed_img).astype(np.uint8))
            img_array.save('./output/'+img_name.split('.')[0]+'_'+str(i+1)+'.png', compress_level=0) # png
            ###

            output = output.clamp_(-1, 1)
            target = target.clamp_(-1, 1)

            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False).item()

            ciede_output = ciede2000(output.cpu().squeeze(0).numpy(), target.cpu().squeeze(0).numpy())
            
        PSNR.update(psnr_val)
        SSIM.update(ssim_val)
        CIEDE.update(ciede_output)

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})\t'
              'CIEDE: {ciede.val:.03f} ({ciede.avg:.03f})'
              .format(i, psnr=PSNR, ssim=SSIM, ciede=CIEDE))

        print('Time: ', intermediate_time, 'sec')

        count = i
        
    print(count+1)
    print('Time: ', total_time/(count+1), 'sec')

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = test_Dataset(opt.test_dir)
    
    #test_dataset = test_Dataset('./RSID_100/test/')
    #test_dataset = test_Dataset('./RICE1_100/test/')
    #test_dataset = test_Dataset('./RICE2_100/test/')
    
    #test_dataset = test_Dataset('./SateHaze1K/test_thin/')
    #test_dataset = test_Dataset('./SateHaze1K/test_moderate/')
    #test_dataset = test_Dataset('./SateHaze1K/test_thick/')
    
    
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=2,
                             pin_memory=True,shuffle=False)


    
    model_dir=opt.trained_dir

    # model_dir='./trained/DA_Net_RSID_146_3007_0974_2239.pk'
    
    # model_dir='./trained/DA_Net_RICE1_135_3671_0989_1529.pk'
    # model_dir='./trained/DA_Net_RICE2_114_3730_0953_1768.pk'
    # model_dir='./trained/DA_Net_SateHaze1k_thick_87_2750_0946_3275.pk'
    # model_dir='./trained/DA_Net_SateHaze1k_moderate_83_3149_0977_3528.pk'
    # model_dir='./trained/DA_Net_SateHaze1k_thin_99_3021_0976_2499.pk'
    
    
    ckp=torch.load(model_dir,map_location=device)
    net=DA_Net_t().to(device)

    
    net=nn.DataParallel(net)
    #summary(net, depth=0)
    summary(net, (1,3,64,64), depth=0)

    param_size = 0
    buffer_size = 0
    for param in net.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Size: {:.3f} MB'.format(size_all_mb))

    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
    
    print('{:<30}  {:<8}'.format('Computational complexity (MACs): ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    net.load_state_dict(ckp['model'])
    test(test_loader, net)

if __name__ == '__main__':
    main()
