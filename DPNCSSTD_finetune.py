from dataset import *
from model import *
import time



def DPNCSSTD_finetune(args):
    img, _, sp = loaddata(args.test_image, args.test_image, 'finetune')
    _, band = sp.shape
    patch_sample, spectral_sample, label = get_samples_all(img, sp, args.windowsize,
                                                           args.method, args.threshold,
                                                           args.show, args.number)
    print('  patch_sample  : ', patch_sample.shape)
    print('spectral_sample : ', spectral_sample.shape)

    finetune_dataset = DPNCSSTD_dataset(patch_sample, spectral_sample, label, 'train')
    finetune_dataloader = torch.utils.data.DataLoader(finetune_dataset, batch_size=patch_sample.shape[0],
                                                      shuffle=True)

    encoder = DPN(band).to(device)
    model = SimCLRStage2(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)

    if args.usepretrain:
        checkpoint = torch.load('./pth/DPNCSSTD_pretrain/pre_Sandiego.pth', map_location=device)
        del checkpoint['model_state_dict']['model.conv16.weight']
        # del checkpoint['model_state_dict']['model.conv16.bias']
        del checkpoint['model_state_dict']['model.conv21.weight']
        # del checkpoint['model_state_dict']['model.conv21.bias']

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        for i, p in enumerate(model.parameters()):
            if 0 < i < 8 or 11 < i < 18:
                p.requires_grad = False

        create_folder(args.finetune_savepath)
        path = args.finetune_savepath + str(args.test_image) + '_DPNCSSTD.pth'

    criterion = torch.nn.BCELoss()
    loss_epoch_list = []

    start = time.time()
    model.train()
    for epoch in range(args.finetune_epoches):

        loss_epoch = 0

        for step, (x, y, labels) in enumerate(finetune_dataloader):
            labels = labels.float().to(device)
            features = model(x.to(device), y.to(device))
            loss = criterion(features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        print(f"Epoch [{epoch + 1}/{args.finetune_epoches}]\t  loss: {loss_epoch} ")
        if len(loss_epoch_list) == 0 or loss_epoch < min(loss_epoch_list):
            print('loss is the minimum,save the model')
            torch.save(model.state_dict(), path)
        loss_epoch_list.append(loss_epoch)

    end = time.time()
    print(f'Finetuning Time:{Fore.BLUE}{end-start}{Style.RESET_ALL}')

    plt.plot(range(1, len(loss_epoch_list) + 1), loss_epoch_list, color='b', label='loss')
    plt.legend(), plt.ylabel('loss value'), plt.xlabel('epochs'), plt.title('Train loss')
    plt.show()
    return end - start
