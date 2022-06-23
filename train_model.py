import torch
import math
def train_model(model, dataloaders, loss, optimizer, save_as = None,save = False, epochs=50, load_wts = False, wts = None, best_model_weights = None, scheduler = None, transform_ = None, max_ = None, min_ = None):
    """ train a model with given params
    Args:
        model: model, extends torch.nn
        dataloaders: dataloader dictionary of the form {"train": dataloader_train_data
                                                        "val": dataloader_val_data
                                                        }
        optimizer: optimization func.
        wts_path: path to torch.nn.Module.load_state_dict for "model"
        epochs: number of epochs to train model
        load_wts: bool true if loading a state dict, false otherwhise


    Return:
        Tuple: model with trained weights and validation training statistics(epoch loss, accuracy)
    """

    #isntantiate validation history, base model waits and loss
    val_loss_history = []
    train_loss_history = []

    val_acc_history = []
    train_acc_history = []

    best_acc = 0.0
    best_optim = None
    #load moadel weigthts
    if load_wts == True:
        print("loading from: "+path_wts)
        checkpoint = torch.load(path_wts)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("acc from prev:{:.4f}".format(checkpoint['best_acc']))

    #train model
    for epoch in range(epochs):
        #import pdb; pdb.set_trace()
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for batch in dataloaders[phase]:
                #send inputs and labels to device
                inputs = batch[0].to(device, non_blocking=True)
                inputs = (inputs - min_)/max_
                labels = batch[1][:,0].to(device, non_blocking=True) # Get only class label
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss for train
                    if phase == 'train':
                        """
                        if transform_ != None:
                            inputs = transform_(inputs)
                            inputs = (inputs - inputs.min())/inputs.max()
                        """
                        preds = model(inputs)
                        loss = lossfun(preds, labels)


                    # Get model outputs and calculate loss for val
                    else:
                        preds = model(inputs)
                        loss = lossfun(preds, labels)

                    #get predictions
                    _, preds = torch.max(preds, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #back propagate loss
                        loss.backward()
                        #update weights
                        optimizer.step()

                    #running statistics
                    running_loss += loss * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            time_elapsed = time.time() - since

            #advance scheduler
            if scheduler != None:
                scheduler.step()

            #update epoch loss and acc
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            #track validation loss and acc
            print('{}: {} epoch_loss: {:.10f} epoch_acc: {:.4f} time: {:.4f}'.format(epoch,phase, epoch_loss, epoch_acc,time_elapsed))


            #update training history
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

            #update best weights
            if phase == 'val' and best_acc < epoch_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optim = copy.deepcopy(optimizer.state_dict())
                best_model_weights= best_model_wts
        #save model
        if epoch ==epochs-1 and save:
            torch.save({
            'best_acc': best_acc,
            'model_state_dict': best_model_wts,
            'optimizer_state_dict': best_optim,
            'best_acc': best_acc,
            }, save_as+"_ep={}.tar".format(epoch))

    val_loss_history = [item.item() for item in val_loss_history]
    val_acc_history = [item.item() for item in val_acc_history ]
    train_loss_history = [item.item() for item in train_loss_history ]
    train_acc_history = [item.item() for item in train_acc_history ]

    model.load_state_dict(best_model_wts)
    print(best_acc)
    history = (val_loss_history, val_acc_history, train_loss_history, train_acc_history, best_acc)
    return model, history


def trainClassifierModel(dataloaders, dropouts, weightdecays, attn_temp = None, device = None, feature_weights = None, epochs=50, scheduler = None, transform_ = None, max_ = None, min_ = None, history_path = ""):
    """ train a model with given params
    Args:
        model: model, extends torch.nn
        dataloaders: dataloader dictionary of the form {"train": dataloader_train_data
                                                        "val": dataloader_val_data
                                                        }
        optimizer: optimization func.
        wts_path: path to torch.nn.Module.load_state_dict for "model"
        epochs: number of epochs to train model
        load_wts: bool true if loading a state dict, false otherwhise


    Return:
        Tuple: model with trained weights and validation training statistics(epoch loss, accuracy)
    """

    numAccTrials = 10 # Number of non increasing epochs before break
    bestOverallAcc = 0.0
    bestModelWeights = None
    bestModelParams = []

    #train model
    for classificationModel_dropout in dropouts: # added
        for classifcationModel_weightDecay in weightdecays: # added
            print(f"train classifier attn_temp:{attn_temp} clf_dropout:{classificationModel_dropout } clf_weightDecay: {classifcationModel_weightDecay}")
            # initialize new model
            patient_model = WaveFusion_contrastive_classifier_Model(device = device, temperature = attn_temp, drop_rate = classificationModel_dropout).to(device)

            #load pretrained weights to wavefusion feature extractor
            patient_model.load_weights(feature_weights)

            patient_model.freeze_parameter_grad(False)

            #specify optimizer and loss function
            optimizer = torch.optim.Adam(patient_model.parameters(), lr=0.0005, weight_decay=classifcationModel_weightDecay)
            lossfun = nn.CrossEntropyLoss()

            val_acc_history = []
            train_acc_history = []

            best_acc = 0.0
            numTrialsCount = 0
            best_model_wts = None
            best_model_params = None

            break_out_flag = False

            for epoch in range(epochs):
                #import pdb; pdb.set_trace()
                since = time.time()

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        patient_model.train()  # Set patient_model to training mode
                    else:
                        patient_model.eval()   # Setpatient_model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    for batch in dataloaders[phase]:
                        #send inputs and labels to device
                        inputs = batch[0].to(device, non_blocking=True)
                        inputs = (inputs - min_)/max_
                        labels = batch[1][:,0].to(device, non_blocking=True) # Get only class label
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            # Get model outputs and calculate loss for train
                            if phase == 'train':
                                """
                                if transform_ != None:
                                    inputs = transform_(inputs)
                                    inputs = (inputs - inputs.min())/inputs.max()
                                """
                                preds = patient_model(inputs)
                                loss = lossfun(preds, labels)

                            # Get model outputs and calculate loss for val
                            else:
                                preds = patient_model(inputs)
                                loss = lossfun(preds, labels)

                            #get predictions
                            _, preds = torch.max(preds, 1)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                #back propagate loss
                                loss.backward()
                                #update weights
                                optimizer.step()

                            #running statistics
                            running_loss += loss * inputs.size(0)
                            running_corrects += torch.sum(preds == labels.data)

                    time_elapsed = time.time() - since


                    #update epoch loss and acc
                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects / len(dataloaders[phase].dataset)

                    #track validation loss and acc
                    print('{}: {} epoch_loss: {:.10f} epoch_acc: {:.4f} time: {:.4f}'.format(epoch,phase, epoch_loss, epoch_acc,time_elapsed))

                    if phase == 'train':
                        #update histories
                        train_acc_history.append(epoch_acc)

                    elif phase == 'val':
                        #update histories
                        val_acc_history.append(epoch_acc)

                        #update best_acc, best_model_weights, reset numTrialsCount
                        if best_acc < epoch_acc:
                            best_acc = epoch_acc
                            best_model_wts = copy.deepcopy(patient_model.state_dict())
                            best_model_params = [classificationModel_dropout, classifcationModel_weightDecay, best_acc]
                            numTrialsCount = 0

                        elif best_acc > epoch_acc:
                            numTrialsCount += 1 #update numTrialsCount

                    # Begin saving results and continuing search
                    if (numTrialsCount >= numAccTrials) or (epoch >= epochs-1):

                        # Append this model's training results to a results lists
                        val_acc_history = [item.item() for item in val_acc_history ]
                        train_acc_history = [item.item() for item in train_acc_history ]

                        #Update Best overall acc. and  overall best model weights:
                        if best_acc > bestOverallAcc:
                            bestOverallAcc = best_acc
                            bestModelWeights = copy.deepcopy(best_model_wts)
                            bestModelParams = copy.deepcopy(best_model_params)
                            del best_model_wts, best_model_params

                        #save results to history .txt file
                        with open(history_path, 'a') as file:
                            file.write(f"\tclassifier params: attn_temp {attn_temp} weightDecay {classifcationModel_weightDecay} dropout {classificationModel_dropout} \n"  )
                            file.write("\t{: <10} {: <10} {:<10} \n".format("epoch ","train acc.","val acc"))
                            for i, data in enumerate(zip(train_acc_history, val_acc_history)):
                                str2 = "\t{: <10} {: <10.3f} {:.3f} \n".format(i+1, data[0], data[1])
                                file.write(str2)
                            acc = float( best_acc )
                            file.write( f"\tbest val acc. { acc } \n" )

                        # reset number of trials counter; continue
                        numTrialsCount = 0
                        break_out_flag = True
                        break


                if break_out_flag:
                    break

            del patient_model
    #Return best model & best model Params
    return bestModelWeights, bestModelParams

def partition_batch(n, neg_other_percent, pat_positive_perc):
  pat_pos = math.ceil(n*pat_positive_perc)
  neg_other = int((n-pat_pos)*neg_other_percent)
  pat_neg = math.floor((n-pat_pos)*(1-neg_other_percent))
  print(pat_pos, neg_other, pat_neg)
  return n, pat_pos, pat_neg
