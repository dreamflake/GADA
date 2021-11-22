import torch

configurations = {
   1: dict( # CurricularFace IR101 Model
        SEED = 1337, # random seed for reproduce results
        DATA_ROOT='G:\\face_dataset',  # the parent root where your train/val/test data are stored
        MODEL_ROOT = 'checkpoint', # the root to buffer your checkpoints
        LOG_ROOT = 'log', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = 'checkpoint/CurricularFace_Backbone.pth', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = "",
        BACKBONE_NAME = 'IR_101', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = "CurricularFace", # support:  ['ArcFace', 'CurricularFace']
        LOSS_NAME = 'Softmax', # support: ['Focal', 'Softmax']
        LFW_THRESHOLD=1.40,
        CFP_THRESHOLD=1.69,
        CPLFW_THRESHOLD=1.68,
        KFACE_THRESHOLD=1.68,
        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 128,
        LR = 0.1, # initial LR
        START_EPOCH = 0, #start epoch
        NUM_EPOCH = 24, # total epoch number
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        RANK = 0,
        GPU = 4, # specify your GPU ids
        DIST_BACKEND = 'nccl',
        DIST_URL = 'tcp://localhost:23456',
        NUM_WORKERS = 4,
        TEST_GPU_ID = [0]
    ),
    2: dict(  # IR50
        SEED=1337,  # random seed for reproduce results
        DATA_ROOT='G:\\face_dataset',  # the parent root where your train/val/test data are stored
        MODEL_ROOT='checkpoint',  # the root to buffer your checkpoints
        LOG_ROOT='log',  # the root to log your train/val status
        BACKBONE_RESUME_ROOT='checkpoint/ms1m-ir50/backbone_ir50_ms1m_epoch120.pth',
        HEAD_RESUME_ROOT="",
        BACKBONE_NAME='IR_50',
        LFW_THRESHOLD=1.40,
        CFP_THRESHOLD=1.67,
        CPLFW_THRESHOLD=1.64,
        KFACE_THRESHOLD=1.60,
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME="ArcFace",  # support:  ['ArcFace', 'CurricularFace']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],

        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=128,
        LR=0.1,  # initial LR
        START_EPOCH=0,  # start epoch
        NUM_EPOCH=24,  # total epoch number
        WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
        MOMENTUM=0.9,
        GPU=4,  # specify your GPU ids
        DIST_BACKEND='nccl',
        DIST_URL='tcp://localhost:23456',
        NUM_WORKERS=4,
        TEST_GPU_ID=[0]
    ),

    3: dict(  # FaceNet
        SEED=1337,  # random seed for reproduce results
        DATA_ROOT='G:\\face_dataset',  # the parent root where your train/val/test data are stored
        MODEL_ROOT='checkpoint',  # the root to buffer your checkpoints
        LOG_ROOT='log',  # the root to log your train/val status
        BACKBONE_RESUME_ROOT='',
        HEAD_RESUME_ROOT="",
        BACKBONE_NAME='FaceNet',
        LFW_THRESHOLD=1.11,
        CFP_THRESHOLD=1.41,
        CPLFW_THRESHOLD=1.33,
        KFACE_THRESHOLD=0.96,
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME="ArcFace",  # support:  ['ArcFace', 'CurricularFace']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[128.0/255.0, 128.0/255.0, 128.0/255.0],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=128,
        LR=0.1,  # initial LR
        START_EPOCH=0,  # start epoch
        NUM_EPOCH=24,  # total epoch number
        WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
        MOMENTUM=0.9,
        GPU=4,  # specify your GPU ids
        DIST_BACKEND='nccl',
        DIST_URL='tcp://localhost:23456',
        NUM_WORKERS=4,
        TEST_GPU_ID=[0]
    ),
}
