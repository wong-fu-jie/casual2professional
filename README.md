# casual2professional
Repository for codes of BT5153 Applied ML Project: Casual2Professional.

Codes provided are all used within a Jupyter notebook environment.

## Understanding the Data:
CycleGAN uses unpaired samples in 2 domains. Therefore the data folder is organised as such, with training and test data split into 2 separate sets, each containing one from domain A(Casual) and domain B(Professional). A further evaluation folder is also kept for final evaluation pictures from domain A.

## Augmenting the Pictures:
Run 'pics_aug.ipynb'. Ensure that image path is pointing to the correct folder.

## Running CycleGAN code:
There a total of 3 separate code files. Use 'c2p_cyclegan.ipynb' if running the model with the data for the first time. The code will save the model after every epoch. Update the epoch_load value in 'c2p_cyclegan_loader.ipynb' to look for the correct saved model to continue training. Finally, the 'c2p_cyclegan_tester.ipynb' code will translate and save all pictures from 'testA' folder, using the model from the file path indicated.

In all codes, ensure that both pic_size and resent_blocks value are consistent with the saved model. For example, if using pic_size = 128 and resnet_blocks = 6 during initial training then the loader must also use the same values when loading the model.

'cyclegan_model.py' keeps all the CycleGAN implementation (to enhance code organisation) and is referenced by all the CycleGAN jupyter files. 