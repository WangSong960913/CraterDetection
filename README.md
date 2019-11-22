# CraterDetection
Crater detection algorithm based on deep learning and semantic segmentation

Our paper is being submitted. We will publish the link after the paper is published.

 Our algorithm contains three main steps. 
Firstly, We need to generate experiment data.Our original moon DEM image can download in https://pan.baidu.com/s/1eSpBLrA-Upqr5qjf6__r8w.  and We random clipe the lunar DEM image to generate data. 
Secondly, using Simple-ResUNet detects crater edges in lunar images
Then, it uses template matching algorithm to compute the position and size of craters. 
Finally, we draw the images of recognized craters and record the location and radius of the craters.

## Dependencies

Our code is based on python3.6. We use Keras [24] as the deep learning framework.

pip install requirement.txt

## Generate Data

python gen_data.py 

train data : set   save_name="./data/train" number=30000 ;valid data : save_name="./data/valid" set  number=3000;test data : save_name="./data/dev" set  number=3000

## Train

python train.py

you can adjust train image number, batch size, epochs and model save path in train.py.If you want to change model, you can change code in model_train.py. (dlinknet,linknet,unet,deep residual net,simple-resunet)

## Detect

python run_extract_craters.py

You should change model_Name value in run_extract_craters.py to chose your trained model. 

## ExperimentResult

The trained model can be downloaded in  https://pan.baidu.com/s/1eSpBLrA-Upqr5qjf6__r8w.

![1574431819461](C:\Users\vs\AppData\Roaming\Typora\typora-user-images\1574431819461.png)

![1574431830956](C:\Users\vs\AppData\Roaming\Typora\typora-user-images\1574431830956.png)

| Model     | Simple-ResUNet | Simple-ResUNet-2 | Ari.S[19]   | Deep   Residual   U-Net [21] |
| --------- | -------------- | ---------------- | ----------- | ---------------------------- |
| Parameter | 23,740,305     | 36,856,401       | 10,278,017  | 23,562,225                   |
| Recall    | **81.16%**     | 80.15%           | 76.10 %     | 76.67%                       |
| Precision | 75.37%         | 77.51%           | **83.16 %** | 77.86%                       |
| F1-Score  | 76.29 %        | 77.07%           | **77.95%**  | 75.59%                       |
| F2-Score  | **78.54%**     | 78.37 %          | 76.49%      | 76.90%                       |

## Authors

**Wang Song**  Email:1140479300@qq.com

**Wei Chao,Zhang Hong,Fu JinWu,Shi Linrui.**

## References

[1]       Ari S , Mohamad A D , Chenchong Z , et al. Lunar Crater Identification via Deep Learning [J]. Icarus, 2018:S0019103518301386.

[2]       Zhang Z , Liu Q ,
Wang Y . Road Extraction by Deep Residual U-Net[J]. IEEE Geoscience and Remote
Sensing Letters, 2018:1-5.

[3]       Ronneberger O ,
Fischer P , Brox T . U-Net: Convolutional Netw-orks for Biomedical Image
Segmentation[J]. Springer International Publishing ,2015.

[4]       Zhou L, Zhang C,
Ming W. D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for
High Resolution Satellite Imagery Road Extraction[C]. 2018 IEEE/CVF Conference
on Computer Vision and Pattern Recognition Workshops (CVPRW). 2018.

## Thanks

Thanks for you star and download. If you have any good opinions or questions, please contact me.Thank you!