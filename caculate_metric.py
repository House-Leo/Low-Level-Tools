'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
import shutil
import lpips


def main():
    # Configurations

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    folder_GT = '/home/yzj6850/houselee/dataset/ntire20/track1-valid-gt' #ntire 2020
    folder_GT = '/home/yzj6850/houselee/dataset/new_realsr/realsr_test/HRx4' #
    # folder_Gen = '/home/yzj6850/houselee/Real-SR/experiments/ntire20/85000'
    # folder_unif = '/home/yzj6850/houselee/ablation_results/align_unif_sub'
    # folder_neg = '/home/yzj6850/houselee/ablation_results/align_neg_sub'
    # folder_align = '/home/yzj6850/houselee/ablation_results/only_align_sub'
    # folder_wsum = '/home/yzj6850/houselee/ablation_results/wsum_sub'

    # folder_select_GT = '/home/yzj6850/houselee/dataset/RealSR/Test/RealESRGAN'
    folder_Gen = '/home/yzj6850/houselee/Real-SR/experiments/real_0725/50000'
    # folder_select_unif = '/home/yzj6850/houselee/ablation_results/select_unif'
    # folder_select_neg = '/home/yzj6850/houselee/ablation_results/select_neg'
    # folder_select_align = '/home/yzj6850/houselee/ablation_results/select_align'
    # folder_select_wsum = '/home/yzj6850/houselee/ablation_results/select_wsum'

    f = open('/home/yzj6850/houselee/dataset/new_realsr/50000.txt','a+')
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()
    crop_border = 4
    suffix = ''  # suffix for Gen images
    test_Y = False  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    # PSNR_all_2 = []
    # PSNR_all_3 = []
    SSIM_all = []
    # SSIM_all_2 = []
    # SSIM_all_3 = []
    LPIPS_all = []
    # LPIPS_all_2 = []
    # LPIPS_all_3 = []
    img_list = sorted(glob.glob(folder_GT + '/*'))

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        #Test LPIPS
        img0 = lpips.im2tensor(lpips.load_image(img_path))
        img1 = lpips.im2tensor(lpips.load_image(os.path.join(folder_Gen, base_name + suffix + '.png')))
        # img2 = lpips.im2tensor(lpips.load_image(os.path.join(folder_neg, base_name + suffix + '.png')))
        # img3 = lpips.im2tensor(lpips.load_image(os.path.join(folder_align, base_name + suffix + '.png')))
        img0 = img0.cuda()
        img1 = img1.cuda()
        # img2 = img2.cuda()
        # img3 = img3.cuda()

        dist01 = loss_fn.forward(img0,img1)
        # dist02 = loss_fn.forward(img0,img2)
        # dist03 = loss_fn.forward(img0,img3)
        # dist01 = loss_fn.forward(img0,img1)

        lp = dist01.cpu().detach().numpy()
        # lp2 = dist02.cpu().detach().numpy()
        # lp3 = dist03.cpu().detach().numpy()

        LPIPS_all.append(lp)
        # LPIPS_all_2.append(lp2)
        # LPIPS_all_3.append(lp3)

        im_GT = cv2.imread(img_path) / 255.
        im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + suffix + '.png')) / 255.
        # im_unif = cv2.imread(os.path.join(folder_unif, base_name + suffix + '.png')) / 255.
        # im_neg = cv2.imread(os.path.join(folder_neg, base_name + suffix + '.png')) / 255.
        # im_align = cv2.imread(os.path.join(folder_align, base_name + suffix + '.png')) / 255.
    #    im_bic = cv2.imread(os.path.join(folder_bic, base_name + suffix + '.png')) / 255.      
        
        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
            # im_unif_in = bgr2ycbcr(im_unif)
            # im_neg_in = bgr2ycbcr(im_neg)
            # im_align_in = bgr2ycbcr(im_align)
#            im_bic_in = bgr2ycbcr(im_bic)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen
            # im_unif_in = im_unif
            # im_neg_in = im_neg
            # im_align_in = im_align
#            im_bic_in = im_bic

        # crop borders
        if im_GT_in.ndim == 3:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
            # cropped_unif = im_unif_in[crop_border:-crop_border, crop_border:-crop_border, :]
            # cropped_neg = im_neg_in[crop_border:-crop_border, crop_border:-crop_border, :]
            # cropped_align = im_align_in[crop_border:-crop_border, crop_border:-crop_border, :]
#            cropped_bic = im_bic_in[crop_border:-crop_border, crop_border:-crop_border, :]
        elif im_GT_in.ndim == 2:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
            # cropped_unif = im_unif_in[crop_border:-crop_border, crop_border:-crop_border, :]
            # cropped_neg = im_neg_in[crop_border:-crop_border, crop_border:-crop_border, :]
            # cropped_align = im_align_in[crop_border:-crop_border, crop_border:-crop_border, :]
#            cropped_bic = im_bic_in[crop_border:-crop_border, crop_border:-crop_border]
        else:
            raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)

        # PSNR2 = calculate_psnr(cropped_GT * 255, cropped_neg * 255)

        # PSNR3 = calculate_psnr(cropped_GT * 255, cropped_align * 255)

        SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)

        # SSIM2 = calculate_ssim(cropped_GT * 255, cropped_neg * 255)

        # SSIM3 = calculate_ssim(cropped_GT * 255, cropped_align * 255)

        #select pred_better and pred_worse patches
        # if lp1[0][0][0][0]<lp2[0][0][0][0] and lp1[0][0][0][0]<lp3[0][0][0][0]:
        # if lp2[0][0][0][0]-lp1[0][0][0][0]>0.05 and lp3[0][0][0][0]-lp1[0][0][0][0]>0.05:
        #     # if len(glob.glob(folder_pred = '*.png')) <= 200:
        #     shutil.copy(folder_GT + '/' + base_name + suffix + '.png', folder_select_GT)
        #     shutil.copy(folder_unif + '/' + base_name + suffix + '.png', folder_select_unif)
        #     shutil.copy(folder_neg + '/' + base_name + suffix + '.png', folder_select_neg)
        #     shutil.copy(folder_align + '/' + base_name + suffix + '.png', folder_select_align)
        #     shutil.copy(folder_wsum + '/' + base_name + suffix + '.png', folder_select_wsum)

        # if PSNR-PSNR2 <= 0.3:
        #     if len(glob.glob(folder_ori = '*.png')) <= 200:
        #         shutil.copy(folder_Gen + '/' + base_name + suffix + '.png', folder_ori)
    #     print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}, \tLPIPS: {:.6f}'.format(
    #         i + 1, base_name, PSNR, SSIM, lp[0][0][0][0]))
    #     f.writelines('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}, \tLPIPS: {:.6f}\n'.format(
    #         i + 1, base_name, PSNR, SSIM, lp[0][0][0][0]))
    #     PSNR_all.append(PSNR)
    #     SSIM_all.append(SSIM)
    # avg_lpips = sum(LPIPS_all)/ len(LPIPS_all)
    # print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}, LPIPS: {:.6f}'.format(
    #     sum(PSNR_all) / len(PSNR_all),
    #     sum(SSIM_all) / len(SSIM_all),
    #     float(avg_lpips[0][0][0][0])))
    # f.writelines('Average: PSNR: {:.6f} dB, SSIM: {:.6f}, LPIPS: {:.6f}'.format(
    #     sum(PSNR_all) / len(PSNR_all),
    #     sum(SSIM_all) / len(SSIM_all),
    #     avg_lpips[0][0][0][0]))
        
        print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}, \tLPIPS: {:.6f}'.format(
            i + 1, base_name, PSNR, SSIM, lp[0][0][0][0]))
        f.writelines('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}, \tLPIPS: {:.6f}\n'.format(
            i + 1, base_name, PSNR, SSIM, lp[0][0][0][0]))
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
    avg_lpips = sum(LPIPS_all)/ len(LPIPS_all)
    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}, LPIPS: {:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all),
        float(avg_lpips[0][0][0][0])))
    f.writelines('Average: PSNR: {:.6f} dB, SSIM: {:.6f}, LPIPS: {:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all),
        avg_lpips[0][0][0][0]))
        
    #     print('{:3d} - {:13}. \tPSNR1: {:.6f} dB, \tSSIM1:{:.6f}, \tLPIPS1: {:.6f}, \tPSNR2: {:.6f}, \tSSIM2:{:.6f}, \tLPIPS2: {:.6f}, \tPSNR3: {:.6f}, \tSSIM3:{:.6f}, \tLPIPS3: {:.6f}'.format(
    #        i + 1, base_name, PSNR, SSIM, lp1[0][0][0][0], PSNR2, SSIM2, lp2[0][0][0][0], PSNR3, SSIM3, lp3[0][0][0][0]))
    #     f.writelines('{:3d} - {:13}. \tPSNR1: {:.6f} dB, \tSSIM1:{:.6f}, \tLPIPS1: {:.6f}, \tPSNR2: {:.6f}, \tSSIM2:{:.6f}, \tLPIPS2: {:.6f}, \tPSNR3: {:.6f}, \tSSIM3:{:.6f}, \tLPIPS3: {:.6f}\n'.format(
    #        i + 1, base_name, PSNR, SSIM, lp1[0][0][0][0], PSNR2, SSIM2, lp2[0][0][0][0], PSNR3, SSIM3, lp3[0][0][0][0]))
    #     PSNR_all.append(PSNR)
    #     PSNR_all_2.append(PSNR2)
    #     PSNR_all_3.append(PSNR3)
    #     SSIM_all.append(SSIM)
    #     SSIM_all_2.append(SSIM2)
    #     SSIM_all_3.append(SSIM3)

    #     # SSIM_all.append(SSIM)
    # avg_lpips = sum(LPIPS_all)/ len(LPIPS_all)
    # avg_lpips_2 = sum(LPIPS_all_2)/ len(LPIPS_all_2)
    # avg_lpips_3 = sum(LPIPS_all_3)/ len(LPIPS_all_3)
    # print('Average: PSNR1: {:.6f} dB, SSIM1: {:.6f}, LPIPS1: {:.6f}, PSNR2: {:.6f}dB, SSIM2: {:.6f}, LPIPS2: {:.6f}, PSNR3: {:.6f}dB, SSIM3: {:.6f}, LPIPS3: {:.6f}'.format(
    #     sum(PSNR_all) / len(PSNR_all),
    #     sum(SSIM_all) / len(SSIM_all),
    #     float(avg_lpips[0][0][0][0]),
    #     sum(PSNR_all_2) / len(PSNR_all_2),
    #     sum(SSIM_all_2) / len(SSIM_all_2),
    #     float(avg_lpips_2[0][0][0][0]),
    #     sum(PSNR_all_3) / len(PSNR_all_3),
    #     sum(SSIM_all_3) / len(SSIM_all_3),
    #     float(avg_lpips_3[0][0][0][0])))
    
    # f.writelines('Average: PSNR1: {:.6f} dB, SSIM1: {:.6f}, LPIPS1: {:.6f}, PSNR2: {:.6f}dB, SSIM2: {:.6f}, LPIPS2: {:.6f}, PSNR3: {:.6f}dB, SSIM3: {:.6f}, LPIPS3: {:.6f}'.format(
    #     sum(PSNR_all) / len(PSNR_all),
    #     sum(SSIM_all) / len(SSIM_all),
    #     avg_lpips[0][0][0][0],
    #     sum(PSNR_all_2) / len(PSNR_all_2),
    #     sum(SSIM_all_2) / len(SSIM_all_2),
    #     avg_lpips_2[0][0][0][0],
    #     sum(PSNR_all_3) / len(PSNR_all_3),
    #     sum(SSIM_all_3) / len(SSIM_all_3),
    #     avg_lpips_3[0][0][0][0]))

    f.close()
        


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

if __name__ == '__main__':
    main()