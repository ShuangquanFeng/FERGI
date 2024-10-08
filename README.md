# FERGI (Facial Expression Reaction to Generated Images)
<div align="center">
    <img src="./images/example_whole_AU4.png" width="300" height="300" alt="Alt text for the image">
</div>

<div align="center">
   <strong>An example of AU4 (brow lowerer) activation in response to a low-quality image generation</strong>
</div>
<br><br>
<div align="center">
    <img src="./images/example_whole_AU12.png" width="300" height="300" alt="Alt text for the image">
</div>

<div align="center">
   <strong>An example of AU12 (lip corner puller) activation in response to a high-quality image generation</strong>
</div>

<br><br>

<div align="center">
    <img src="./images/FERGI_flow_chart.png" alt="Alt text for the image">
</div>

<div align="center">
   <strong>A flow chart summarizing the pipeline our paper proposed</strong>
</div>

<br><br>

Researchers have proposed to use data of human preference feedback to fine-tune text-to-image generative models. However, the scalability of human feedback collection has been limited by its reliance on manual annotation. Therefore, we develop and test a method to automatically score user preferences from their spontaneous facial expression reaction to the generated images. We collect a dataset of Facial Expression Reaction to Generated Images (FERGI) and show that the activations of multiple facial action units (AUs) are highly correlated with user evaluations of the generated images. We develop an FAU-Net (Facial Action Units Neural Network), which receives inputs from an AU estimation model, to automatically score user preferences for text-to-image generation based on their facial expression reactions, which is complementary to the pre-trained scoring models based on the input text prompts and generated images. Integrating our FAU-Net valence score with the pre-trained scoring models improves their consistency with human preferences. This method of automatic annotation with facial expression analysis can be potentially generalized to other generation tasks.

# Getting Started
## Dependencies
The primary dependencies include NumPy, pandas, SciPy, Matplotlib, seaborn, OpenCV, MediaPipe, PyTorch, Torchvision, Trasnformers, [CLIP](https://github.com/openai/CLIP), [BLIP](https://github.com/salesforce/BLIP),  [ImageReward](https://github.com/THUDM/ImageReward), and [HPS v2](https://github.com/tgxs002/HPSv2).

## Datasets
### AU Datasets
The datasets for training the AU models [DISFA](http://mohammadmahoor.com/disfa/) and [DISFA+](http://mohammadmahoor.com/disfa/) are supposed to be stored at "../FER_datasets/DISFA" and "../FER_datasets/DISFAPlus" respectively (paths specificed in config.py).

### FERGI Dataset
<strong><ins>FERGI dataset is available for research purposes. Please request it by filling out this [form](https://forms.gle/ja1DUNumBnGSkMMC8).</ins></strong> The dataset is supposed to be stored in the "data" folder. Although the raw dataset is not provided in the github repository, the processed facial features of the videos in the dataset has already been provided in the "data" folder.

## Pretrained Models
Multiple pretrained models are used in our model training and analysis. They need to be downloaded from the following links and stored in the "pretrained_models" folder.

The face recognition model used as the pretrained model to fine-tune for training the AU recognition model can be downloaded [here](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215650&parId=4A83B6B633B029CC%215581&o=OneUp). The download link is provided in the [github repository of InsightFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo). It is supposed to be renamed as "glint360k_cosface_r50_fp16_0.1.pth" and stored in the "pretrained_models" folder after being downloaded.

The face detection model can be downloaded [here](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite). The download link is provided in the [official document of MediaPipe](https://developers.google.com/mediapipe/solutions/vision/face_detector).

The facial landmark detection model can be downloaded [here](https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view). The download link is provided in the [github repository of pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark).
## AU Datasets Preprocessing
Run preprocess_DISFA.py and preprocess_DISFAPlus.py for preprocessing the AU datasets.

## AU Model Training
Run DISFAwithPlus_train_model.py for the training AU recognition model. The trained AU models will be saved in the folder "AU_models". The AU model used for following analysis can also be downloaded [here](https://drive.google.com/file/d/14Y5h-l6FurSdYBhhH4MaJo7VXsbIEhyC/view?usp=drive_link). <strong>Note that this model is trained on DISFA and DISFA+ and thus should be used for research purposes only. If you use this model in your paper, you should also cite the papers of DISFA and DISFA+ (see the terms of use for [DISFA](http://mohammadmahoor.com/disfa-contact-form/) and [DISFA+](http://mohammadmahoor.com/disfa-plus-request-form/)) in addition to our paper.</strong>

## Facial Feature Processing
Run clips_facial_process.py for processing the facial features of the videos in the FERGI dataset. The results from our model has already been provided in the "data" folder.

## FERGI Dataset Preprocessing
Run preprocess_image_data.py, preprocess_baseline_data.py, and preprocess_reaction_data.py for preprocessing the data of generated images, the data of baseline videos, and the data of reaction videos in the FERGI dataset respectively. The results are saved in the "preparation" folder.

## Image Preference Classification
Run image_preference_binary_classification_based_on_ranking_NN.py for binary classification of image preferences (Section 6.2 in the paper). The results are saved in the "results" folder.

## Result Analysis
Run result_analysis.ipynb for analyzing and visualizing the results (Sections 6.1 and 6.2 in the paper). The visualizations are saved in the "figures" folder.

## Citation

```
@article{feng2023fergi,
  title={FERGI: Automatic Annotation of User Preferences for Text-to-Image Generation from Spontaneous Facial Expression Reaction},
  author={Feng, Shuangquan and Ma, Junhua and de Sa, Virginia R},
  journal={arXiv preprint arXiv:2312.03187},
  year={2023}
}
```