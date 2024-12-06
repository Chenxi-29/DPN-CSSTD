from dataset import *
from model import *


def DPNCSSTD_detection(args):
    img, gt, sp = loaddata(args.test_image, args.test_image, 'detection')
    h, w, band = img.shape
    patch_sample, spectral_sample = get_patch_data(img)

    if args.usepretrain:
        path = args.finetune_savepath + str(args.test_image) + '_DPNCSSTD.pth'

    encoder = DPN(band).to(device)
    model = SimCLRStage2(encoder).to(device)

    model.load_state_dict(torch.load(path), strict=True)

    detection_dataset = DPNCSSTD_dataset(patch_sample, spectral_sample, 1, 'test')
    detection_datloader = torch.utils.data.DataLoader(detection_dataset, batch_size=1000, shuffle=False)
    target_detector = np.zeros((1, 1))

    start = time.time()
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(detection_datloader):
            result = model(x.to(device), y.to(device))
            target_detector = np.concatenate((target_detector, result.cpu()))
    target_detector = target_detector[1:, :]
    end = time.time()

    target_detector = np.array(target_detector, dtype=np.float32)
    detection = target_detector.reshape(h, w)
    plt.figure(), plt.imshow(detection), plt.show()

    create_folder(args.detection_savepath)
    if args.usepretrain:
        sio.savemat(args.detection_savepath + str(args.test_image) + '_DPNCSSTD.mat', {'detection': result})

    auc1, auc2, auc3 = plot_roc_curve(detection, gt.reshape(h, w), 0)

    print(f'Detection Time:{Fore.BLUE}{end-start}{Style.RESET_ALL}')

    return auc1, auc2, auc3, end - start, detection
