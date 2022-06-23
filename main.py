import torchvision
import torch
import os
import time
import torch.optim as optim
from transforms import AddPinkNoise, DropInput, AddGaussianNoise, GaussianCorruption, TwoCropTransform
from dataset import Motion_Dataset_Patient, motion_collate, standard_motion_dataloader, motion_dataloader, BalancedBatchSampler
from loss import Patient_SupConLoss
from models import attention1d, Wave_Lead_Conv, WaveFusion_Feature_Model, AttrProxy, WaveFusion_contrastive_classifier_Model
from train_model import train_model, trainClassifierModel, partition_batch


if __name__ == "__main__":
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    threeD = False # Set to true if training 3DCNN
    save = False # If true save model
    save_as = '' #directory to save files. ~/NAME_OF_File
    load_wts = False # If true load pretrained model weights
    path_wts = '' # Path to weights for loading pretrained model
    
    # Fixed parameters
    num_workers = 0
    learning_rate = 0.05 # learning rate
    bsz = 500 # patient batch size
    momentum = 0.0005 # momentum
    epochs = 25 # contrastive epochs
    epoch_patient = 150 # Classification task epochs
    pos_pat_perc = 0.5
    momentum = 0.0005 # momentum
    
    data_dir = "/data_avg_25_stft_post"  # path to data files
    history_path = "/results.txt" # Path to log
    save_path = "" # Directory to save .pth files

    #Pamrameter search all params
    f = open(history_path, "w")
    f.write('Supcon Model training Log')
    f.close()


    train_transform =  torch.nn.Sequential(
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomResizedCrop((39,11),scale=(0.5, 1.0)),
            #transforms.GaussianBlur(kernel_size=(5,3), sigma=(0.1, 0.1))
            AddPinkNoise(mean = 0, std = 0.85, p=0.9, device_=device),
            DropInput(p = 0.4, x = 0.5, device_ = device),
            GaussianCorruption(mean = 0, std = 0.02, p = 0.5, x = 0.5, device_ = device),
        )

    motion_datasets = {x: Motion_Dataset_Patient(os.path.join(data_dir, x),threeD = threeD, transform=None) for x in ['train', 'val']}
    for other_neg_perc in [0.0]: #### Split by other_neg_perc

        bestModels = [] #One best model for each batch size
        for batch_size in [500,1000,2000]:
            bestBszAcc = 0.0
            bestModelWeights = None
            bestModelTag = []  #contains params for best model
            for attn_temp in [27.5, 32.5, 37.5]: #added  [27.5, 32.5, 37.5]
                #for epochs in [10, 25]:
                for supcon_temp  in [0.01, 0.05, 0.1, 0.25]:
                    for embedModel_weightDecay in [0.001, 0.005, 0.0075 ]:


                        print("batch_size", batch_size, "other_neg_perc", other_neg_perc, "pos_pat_perc", pos_pat_perc)
                        balanced_batch_sampler = BalancedBatchSampler(motion_datasets['train'], num_pat_spilt = partition_batch(batch_size, other_neg_perc, pos_pat_perc))

                        TwoCropTransform_ = TwoCropTransform(train_transform, device)
                        max_ = float(motion_datasets['train'].the_data.max())
                        min_ = float(motion_datasets['train'].the_data.min())

                        results_overall = {}

                        train_loader =  motion_dataloader(motion_datasets['train'], num_workers = 0, sampler = balanced_batch_sampler, pin_memory = True)
                        
                        model = WaveFusion_Feature_Model(device = device, temperature = attn_temp).to(device)
                        
                        criterion = Patient_SupConLoss(temperature=supcon_temp,base_temperature=1)
                        
                        contrastive_loss_log = []
                        
                        optimizer = optim.SGD(model.parameters(),
                                    lr = learning_rate,
                                    momentum = momentum,
                                    weight_decay = embedModel_weightDecay)

                        print(f"train contrastive model... params batch_size:{batch_size} attn_temp:{attn_temp} supcon_tempc:{supcon_temp} embedModel_weightDecay:{embedModel_weightDecay}")
                        
                        
                        model.train()

                        #training routine
                        for epoch in range(0, epochs):
                            time1 = time.time()
                            for idx, (images, labels) in enumerate(train_loader):
                                images = images.to(device, non_blocking=True)
                                images = TwoCropTransform_(images)
                                images = torch.cat([(images[0]-images[0].min())/images[0].max(), (images[1]-images[1].min())/images[1].max()], dim=0)
                                labels = labels.to(device, non_blocking=True)
                                bsz = labels.shape[0]

                                # compute loss
                                features = model(images)
                                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                                loss = criterion(features, labels)
                                # SGD
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                            contrastive_loss_log.append(loss)

                            print('epoch {}, avg loss {:.5f} total time {:.3f}'.format(epoch+1, loss, time.time()-time1 ))
                        
                        contrastive_loss_log = [item.item() for item in contrastive_loss_log]
                        best_model_wts = copy.deepcopy(model.state_dict())

                        #write results to history file
                        with open(history_path, 'a') as file:
                            file.write("contrastive training loss log: \n")
                            file.write(f"attn_temp {attn_temp}, supcon_temp {supcon_temp}, embedModel_weightDecay {embedModel_weightDecay} \n")
                            for i, item in enumerate(contrastive_loss_log):
                                str1 = f"epoch {i+1} {loss} \n"
                                file.write(str1)

                        # Free memory
                        del model, criterion, optimizer, contrastive_loss_log, train_loader, balanced_batch_sampler, TwoCropTransform_
                        
                        classificationModel_dropout = [0.5, 0.67, 0.75]
                        classifcationModel_weightDecay = [0.001, 0.005, 0.0075 ]

                        #Train classifier model

                        #create dataloaders
                        train_loader =  standard_motion_dataloader(motion_datasets['train'], bsz, num_workers = 0, pin_memory = True)
                        val_loader =  standard_motion_dataloader(motion_datasets['val'], bsz, num_workers = 0, pin_memory = True)

                        data_loaders_dict = {"train": train_loader, "val": val_loader}

                        #train patient specific model model
                        clf_model_wts, best_params = trainClassifierModel(
                                                            dataloaders = data_loaders_dict,
                                                            feature_weights = best_model_wts,
                                                            epochs =epoch_patient,
                                                            device = device,
                                                            scheduler = None,
                                                            attn_temp = attn_temp,
                                                            transform_ = train_transform,
                                                            dropouts = classificationModel_dropout,
                                                            weightdecays = classifcationModel_weightDecay,
                                                            history_path = history_path,
                                                            max_ = max_,
                                                            min_ = min_)
                        if bestBszAcc < best_params[2]:
                            bestBszAcc = best_params[2]
                            bestModelWeights = copy.deepcopy(clf_model_wts)
                            bestModelTag = best_params
                            bestModelTag.extend([embedModel_weightDecay, supcon_temp, attn_temp, batch_size, other_neg_perc])


                        del clf_model_wts, best_params
                    
            bestModels.append((bestModelTag, bestModelWeights))

    for params, model in bestModels:
        fname = f"WaveFusion_Param_:{params[7]}_batchSize:{params[6]}_ACC:{params[2]}_embeWd:{params[3]}_supConTemp:{params[4]}_attnTemp:{params[5]}_clfDp:{params[0]}_clfWd:{params[1]}"
        torch.save(model, save_path+"/"+fname)
