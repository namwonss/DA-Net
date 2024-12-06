from data_utils import *
from torch.backends import cudnn
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
from tqdm import tqdm
from DA_Net import DA_Net_t
from torchvision.models import vgg16
from perceptual import LossNetwork
from torch.utils.data import DataLoader

def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr

def train(train_model, train_loader, optim, criterion, loss_network, scaler):
    lambda_loss = 0.04
    torch.cuda.empty_cache()
    losses = []    
    tbar = tqdm(train_loader)
    train_model.train()
    for batch in tbar:
        x = batch[0].to(opt.device) # hazy
        y = batch[1].to(opt.device) # GT
        with autocast():
            out = train_model(x)

            pixel_loss = criterion[0](out, y)
            perceptual_loss = loss_network(out, y)

            loss = pixel_loss + lambda_loss*perceptual_loss

            tbar.set_description("Total Loss: {:.5f}, L1_loss: {:.5f}, perceptual_loss : {:.5f}".format(loss, pixel_loss, lambda_loss*perceptual_loss))
            
            if opt.clip:
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 0.2)
        optim.step()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for param in train_model.parameters():
            param.grad = None
        losses.append(loss.item())


def test(test_model, loader_test):
    test_model.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        pred = test_model(inputs)
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        return np.mean(ssims), np.mean(psnrs)
    
if __name__ == "__main__":
    seed_torch(seed=1234)

    # dataset
    BS = opt.bs
    print(BS)
    path = opt.dataset_dir

    trainpath = opt.train
    testpath = opt.test

    loader_train = DataLoader(dataset=RS_Dataset(path+'/'+trainpath,train=True,format='.png'),batch_size=BS,shuffle=True)
    loader_test = DataLoader(dataset=RS_Dataset(path+'/'+testpath,train=False,size='whole img',format='.png'),batch_size=1,shuffle=False)

    net = DA_Net_t()
    net = net.to(opt.device)
    pytorch_total_params = sum(p.nelement() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params / 1e6))
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = [nn.L1Loss().to(opt.device)]

    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(opt.device)
    for param in vgg_model.parameters():
        param.requires_grad = False

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    
    scaler = GradScaler()

    optimizer = torch.optim.AdamW(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr * 1e-2)

    summary(net, depth=5)

    max_ssim = 0
    max_psnr = 0

    for epoch in tqdm(range(opt.epochs + 1)):
        torch.cuda.empty_cache()
        train(net, loader_train, optimizer, criterion, loss_network, scaler)

        scheduler.step()

        if epoch % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)
                print(f'\nepoch :{epoch} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'model': net.state_dict()
                }, os.path.join(opt.model_dir, 'DA-Net'+'_'+'RSID'+'_'+str(epoch)+'.pk'))
                print(f'\n model saved at epoch :{epoch}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

                # RSID
                # RICE1
                # RICE2
                # SateHaze1k_thin
                # SateHaze1k_moderate
                # SateHaze1k_thick

