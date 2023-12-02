# FERGI (Facial Expression Reaction to Generated Images)
<div align="center">
    <img src="./images/example.png" width="500" height="500" alt="Alt text for the image">
</div>

<div align="center">
   <strong>An example of AU4 activation in response to a low-quality image generation</strong>
</div>

<div align="center">
    <img src="./images/FERGI_flow_chart.png" alt="Alt text for the image">
</div>

Researchers have proposed to use data of human preference feedback to fine-tune text-to-image generative models. However, the scalability of human feedback collection has been limited by its reliance on manual annotation. Therefore, we develop and test a method to automatically annotate user preferences from their spontaneous facial expression reaction to the generated images. We collect a dataset of Facial Expression Reaction to Generated Images (FERGI) and show that the activations of multiple facial action units (AUs) are highly correlated with user evaluations of the generated images. Specifically, AU4 (brow lowerer) is most consistently reflective of negative evaluations of the generated image. This can be useful in two ways. Firstly, we can automatically annotate user preferences between image pairs with substantial difference in AU4 responses to them with an accuracy significantly outperforming state-of-the-art scoring models. Secondly, directly integrating the AU4 responses with the scoring models improves their consistency with human preferences. Additionally, the AU4 response best reflects the user's evaluation of the image fidelity, making it complementary to the state-of-the-art scoring models, which are generally better at reflecting image-text alignment. Finally, this method of automatic annotation with facial expression analysis we demonstrated can be potentially generalized to other generation tasks.

# Getting Started
## Dependencies
The primary dependencies include NumPy, pandas, SciPy, Matplotlib, seaborn, OpenCV, MediaPipe, PyTorch, Torchvision, Trasnformers, [CLIP](https://github.com/openai/CLIP), [BLIP](https://github.com/salesforce/BLIP),  [ImageReward](https://github.com/THUDM/ImageReward), and [HPS v2](https://github.com/tgxs002/HPSv2).
## Datasets
The datasets for training the AU models [DISFA](http://mohammadmahoor.com/disfa/) and [DISFA+](http://mohammadmahoor.com/disfa/) are supposed to be stored at "../FER_datasets/DISFA" and "../FER_datasets/DISFAPlus" respectively (paths specificed in config.py).

FERGI dataset is available for research purposes. Please request it by filling out this [form](https://forms.gle/ja1DUNumBnGSkMMC8). The dataset is supposed to be stored in the "data" folder.
## AU Datasets Preprocessing
Run preprocess_DISFA.py and preprocess_DISFAPlus.py for preprocessing the AU datasets.

## AU Model Training
Run DISFAwithPlus_train_model.py for training AU models. The trained AU models will be saved in the folder "AU_models". 
