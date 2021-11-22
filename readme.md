
# Geometrically Adaptive Dictionary Attack on Face Recognition
This is the Pytorch code of our paper "Geometrically Adaptive Dictionary Attack on Face Recognition" (WACV2022). 


# Getting started

## Dependencies
The code of GADA uses various packages such as Python 3.7, Pytorch 1.6.0, cython=0.29.21, and it is easy to install them by copying the existing environment to the current system to install them easily. 

We have saved the conda environment for both Windows and Ubuntu, and you can copy the conda environment to the current system.
You can install the conda environment by entering the following command at the conda prompt.

> conda env create -f GADA_ubuntu.yml

After setting the environment, you may need to compile the 3D renderer by entering the command.

At the '_3DDFA_V2\Sim3DR' path

> python setup.py build_ext --inplace

Since 3D Renderer has already been compiled on Windows and Ubuntu, there may be no problem in running the experiment without executing the above command.

## Pretrained face recognition models
You can download the pretrained face recogntion models from [face.evoLVe](https://drive.google.com/drive/folders/1omzvXV_djVIW2A7I09DWMe9JR-9o_MYh) and [CurricularFace](https://github.com/HuangYG123/CurricularFace)

After downloading the checkpoint files, place 'backbone_ir50_ms1m_epoch120.pth' into '/checkpoint/ms1m-ir50/' and 'CurricularFace_Backbone.pth' into '/checkpoint/'



## Dataset
You can download test image sequences for the LFW and CPLFW datasets from the following links.

[LFW test image sequence](https://drive.google.com/file/d/1vbfS168AGiBCybxB0IUaMq6xsOtu5J75/view?usp=sharing)

[CPLFW test image sequence](https://drive.google.com/file/d/1CUtUoTkiILaogfxnqTAVsSPa_txS49p6/view?usp=sharing)

Place them into the root folder of the project.

Each image sequence has 500 image pairs for dodging and impersonation attack.

These images are curated from the aligned face datasets provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe).

## Usage

You can perform an attack experiment by entering the following command.

> python attack.py --model=2 --attack=EAGD --dataset=LFW

The model argument is the index of the target facial recognition model.
> 1: CurricularFace ResNet-100, 2: ArcFace ResNet-50, 3: FaceNet

The attack argument indicates the attack method.
> HSJA, SO, EA, EAD, EAG, EAGD, EAG, EAGDR, EAGDO, SFA, SFAD, SFAG, SFAGD

If --targeted is given as an execution argument, impersonation attack is performed. If no argument is given, dodging attack is performed by default.

The dataset argument sets which test dataset to use and supports LFW and CPLFW.

If you want to enable stateful detection as a defense, pass the --defense=SD argument to the command line.


When an experiment is completed for 500 test images,
a 'Dataset_NumImages_targeted_attackname_targetmodel_defense_.pth' file is created in the results folder like 'CPLFW_500_1_EVGD_IR_50_gaussian_.pth'.

Using plotter.py, you can load the above saved file and print various results, such as the l2 norm of perturbation at 1000, 2000, 5000, and 10000 steps, the average number of queries until the l2 norm of perturbation becomes 2 or 4, adversarial examples, etc.



# Citation
If you find this work useful, please consider citing our paper :)
We provide a BibTeX entry of our paper below:

```
    @article{byun2021geometrically,
    title={Geometrically Adaptive Dictionary Attack on Face Recognition},
    author={Byun, Junyoung and Go, Hyojun and Kim, Changick},
    journal={arXiv preprint arXiv:2111.04371},
    year={2021}
    }
```

## Acknowledgement

* The 3DDFA_V2 module is modified from [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2).
* The HSJA attack module is modified from [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox).
* The Sign-OPT attack module is modified from [Attackbox](https://github.com/cmhcbb/attackbox).
* The EA attack module is modified from [ARES (Adversarial Robustness Evaluation for Safety)](https://github.com/thu-ml/ares/blob/main/ares/attack/evolutionary_worker.py).
* The SFA attack module is modified from [Sign-Flip-Attack](https://github.com/cwllenny/Sign-Flip-Attack).
* The LPIPS module is modified from [Perceptual Similarity Metric and Dataset](https://github.com/richzhang/PerceptualSimilarity).

* The CurricularFace model is borrowed from [CurricularFace](https://github.com/HuangYG123/CurricularFace).
* The ArcFace model and the LFW and CPLFW datasets are downloaded from [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe).
* The FaceNet model is borrowed from [Face Recognition Using Pytorch](https://github.com/timesler/facenet-pytorch).


