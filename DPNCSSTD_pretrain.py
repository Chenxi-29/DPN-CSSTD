from __future__ import print_function
from tqdm import tqdm
from model import DPN, SimCLRStage1, NT_Xent_loss
from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def DPNCSSTD_pretrain(args):
    windowSize = args.windowsize
    epochs = args.pre_epoches
    BATCH_SIZE = args.pre_batchsize
    learning_rate = args.pre_lr
    path = args.pre_savepath
    create_folder(path)

    # ——————————————  Data augmentation  ——————————————
    preimg = sio.loadmat('./data/Sandiego.mat')['data']
    h, w, band = preimg.shape
    patchdata, spectraldata = add(preimg, windowSize)
    print(f'patch size{patchdata.shape},spectra sample:{spectraldata.shape}')

    aug_data = HSIDataset_train_unite(patchdata, spectraldata,
                                      aug1=augment_unite_randomflip, aug2=augment_spectral)
    pre_dataloader = DataLoader(aug_data, batch_size=BATCH_SIZE, shuffle=True)

    # ——————————————  Pretraining ——————————————
    encoder = DPN(band).to(device)
    model = SimCLRStage1(encoder).to(device)
    Loss_feature = NT_Xent_loss(0.9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_epoch_list = []
    model.train()
    start = time.time()

    for epoch in range(epochs):
        loss_epoch = 0
        for step, (x, y) in enumerate(tqdm(pre_dataloader)):
            x = tuple(x)
            y = tuple(y)
            x_i, x_j = x[0], x[1]
            y_i, y_j = y[0], y[1]
            z_i, z_j = model(x_i.to(device), x_j.to(device), y_i.to(device), y_j.to(device))

            loss = Loss_feature(z_i, z_j)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}]\t  loss: {loss_epoch} ")
        if len(loss_epoch_list) == 0 or loss_epoch < min(loss_epoch_list):
            print('loss is the minimum,save the model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dice': optimizer.state_dict(),
            }, args.pre_savepath +'DPNCSSTD_pretrain.pth')
        loss_epoch_list.append(loss_epoch)

    end = time.time()
    print('==> Saving...')
    print('Pretraining time:', end - start)

    plt.plot(range(1, len(loss_epoch_list) + 1), loss_epoch_list, color='b', label='loss')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Loss'), plt.show()


