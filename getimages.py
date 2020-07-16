import serialize
import os
import os
import serialize
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def saveResultasImage(hundred_all, fiftyfive_all, thirty_all):
    hundred_path = hundred_all[0][0]
    fiftyfive_path = fiftyfive_all[0][0]
    thirty_path = thirty_all[0][0]
    filename = hundred_path.split(os.sep)[-5].upper() + " " + hundred_path.split(os.sep)[-4].upper() 
    hundred = pd.read_csv(hundred_path)
    fiftyfive = pd.read_csv(fiftyfive_path)
    thirty = pd.read_csv(thirty_path)

    hundred_pre = hundred_all[1][0].split(os.sep)[-1].split("_")[0]
    fiftyfive_pre = fiftyfive_all[1][0].split(os.sep)[-1].split("_")[0]
    thirty_pre = thirty_all[1][0].split(os.sep)[-1].split("_")[0]

    f,ax1 = plt.subplots(2,2,figsize =(15,10))
    f.suptitle(filename, fontsize=16)
    a1 = sns.pointplot(x='epochs',y='trainloss', data=hundred, ax=ax1[0][0])
    a = sns.pointplot(x='epochs',y='testloss', color='red',data=hundred, ax=ax1[0][0])
    a1.set(xlabel='Epochs', ylabel='Loss')
    a1.set_title(hundred_path.split(os.sep)[-3] + " frames")

    a2 = sns.pointplot(x='epochs',y='trainloss', data=fiftyfive, ax=ax1[0][1])
    a = sns.pointplot(x='epochs',y='testloss', color='red',data=fiftyfive, ax=ax1[0][1])
    a2.set(xlabel='Epochs', ylabel='Loss')
    a2.set_title(fiftyfive_path.split(os.sep)[-3] + " frames")

    a3 = sns.pointplot(x='epochs',y='trainloss', data=thirty, ax=ax1[1][0])
    a = sns.pointplot(x='epochs',y='testloss', color='red',data=thirty, ax=ax1[1][0])
    a3.set(xlabel='Epochs', ylabel='Loss')
    a3.set_title(thirty_path.split(os.sep)[-3] + " frames")
    ax1[-1, -1].axis('off')
    test_patch = mpatches.Patch(color='red', label='Test')
    train_patch = mpatches.Patch(color='blue', label='Train')
    plt.legend(handles=[train_patch, test_patch])
    
    plt.text(0,0.3,'Validation ',color='black',fontsize = 18)
    plt.text(0,0.2,hundred_path.split(os.sep)[-3] + ' Frames: ' + hundred_pre,color='black',fontsize = 18)
    plt.text(0,0.1,'55 Frames:  ' + fiftyfive_pre,color='black',fontsize = 18)
    plt.text(0,0,'30 Frames:  ' + thirty_pre,color='black',fontsize = 18)
   

    plt.show()
    outputfolder = "resultimage"
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    save_path = os.path.join(outputfolder,  filename + ".png")
    plt.savefig(save_path)
    plt.clf()

modules = ["FFNN", "CNN2DLSTM", "CNN3D", "TWOSTREAM", "POSITIONFFNN", "POSITIONLSTM"]
trajectories = [False, False, False, False, True, True]
frames = [100, 55, 30]
backgrounds = [True, False, None]
results_files_name = serialize.get_all_results_names()
predictions_files_name = serialize.get_all_predictions_names()

def getnetworkresults(listoffiles, module, trajectory, num_frames, background, typeresult):
    files = []
    backgroundpath = "trajectory"
    if trajectory == False:
        if background == True:
            backgroundpath = "background"
        elif background == False:
            backgroundpath = "no_background"
    for path in listoffiles:
        pathsplit = path.split(os.sep)
        if backgroundpath in pathsplit and str(num_frames) in pathsplit and module in pathsplit and typeresult in pathsplit:
            files.append(path)
    if len(files) == 0:
        raise FileNotFoundError("No data were found.")
    return files
i = 1
one_network = []
for background in backgrounds:
    for module, trajectory in zip(modules, trajectories):
        for frame in frames:
            if (trajectory == True and frame == 100) or (background == False and frame == 100):
                frame = 99
            if background is not None and trajectory == True:
                continue
            if background is None and trajectory == False:
                continue
            net = getnetworkresults(results_files_name, module, trajectory, frame, background, "results")
            pre = getnetworkresults(predictions_files_name, module, trajectory, frame, background, "predictions")
            one_network.append([net, pre])
            if len(one_network) == 3 and i == 1:
                saveResultasImage(one_network[0], one_network[1], one_network[2])
                one_network = []
                i = 2