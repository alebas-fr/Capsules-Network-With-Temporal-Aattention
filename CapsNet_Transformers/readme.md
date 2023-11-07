# Hyperparameters

The hyperparameters of the models are in a json file in the hyperparameters folder.

The important hyperparameters of the backbone correspond to : 
- **backbone**, CNN architectures use capsule networks before. Only resnet and efficient-net are supported.
- **input_dim**, The input dimensions of the capsule network, i.e. the output dimensions of the backbone (512 for resnet and  1536 for efficient-net) .
- **layers to unfreeze**, the layers of the backbone that we want to unfreeze for fine-tuning. 
- **layers to unfreeze**, This hyperparameter allows to unfreeze the last two layers of the resnet backbone which correspond to the average pooling and the classifier. 

For the other parameters, please refer to the json file. 
