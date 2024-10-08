Segmentation using UNeT 
Dataset: OxfordIIITPet

Network architecture:
    UNet
    Contracting/encoder layers no_of_ch: (3->64->128->256->512->1024)                   
    Expansive/decoder layers ch: (1024->512->256->256->128->64->3)
    All RelU non-linearity except Sigmoid for output layer

Results:

![alt text](https://github.com/ferozalitm/Segmentation_using_UNet/blob/main/Results/train_image_mask_pred_segmnt_concate-149-0.png)
![alt text](https://github.com/ferozalitm/Segmentation_using_UNet/blob/main/Results/train_image_mask_pred_segmnt_concate-149-0.png)
![alt text](https://github.com/ferozalitm/Segmentation_using_UNet/blob/main/Results/train_image_mask_pred_segmnt_concate-149-0.png)
