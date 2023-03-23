from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import torch, time
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error

# https://www.cnblogs.com/ysugyl/p/8711205.html
# iris = load_iris()
# X_trainval,X_test,y_trainval,y_test = train_test_split(iris.data,iris.target,random_state=0)
# X_train,X_val,y_train,y_val = train_test_split(X_trainval,y_trainval,random_state=1)
# print("Size of training set:{} size of validation set:{} size of teseting set:{}".format(X_train.shape[0],X_val.shape[0],X_test.shape[0]))
#

seed = 163

# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class data_loader():
    """custom data loader for batch training

    """

    def __init__(self, path_viscosity, path_raman, path_density, path_ri, device, scaling=False):
        """
        Inputs
        ------
        path_viscosity : string
            path for the viscosity HDF5 dataset

        path_raman : string
            path for the Raman spectra HDF5 dataset

        path_density : string
            path for the density HDF5 dataset

        path_ri : String
            path for the refractive index HDF5 dataset

        device : CUDA

        scaling : False or True
            Scales the input chemical composition.
            WARNING : Does not work currently as this is a relic of testing this effect,
            but we chose not to scale inputs and Cp are calculated with unscaled values in the network."""

        f = h5py.File(path_viscosity, 'r')

        # List all groups
        self.X_columns = f['X_columns'][()]

        # Entropy dataset
        X_entropy_train = f["X_entropy_train"][()]
        y_entropy_train = f["y_entropy_train"][()]

        X_entropy_valid = f["X_entropy_valid"][()]
        y_entropy_valid = f["y_entropy_valid"][()]

        X_entropy_test = f["X_entropy_test"][()]
        y_entropy_test = f["y_entropy_test"][()]

        # Viscosity dataset
        X_train = f["X_train"][()]
        y_train = f["y_train"][()]

        X_valid = f["X_valid"][()]
        y_valid = f["y_valid"][()]

        X_test = f["X_test"][()]
        y_test = f["y_test"][()]

        # Tg dataset
        X_tg_train = f["X_tg_train"][()]
        X_tg_valid = f["X_tg_valid"][()]
        X_tg_test = f["X_tg_test"][()]

        y_tg_train = f["y_tg_train"][()]
        y_tg_valid = f["y_tg_valid"][()]
        y_tg_test = f["y_tg_test"][()]

        f.close()

        # Raman dataset
        f = h5py.File(path_raman, 'r')
        X_raman_train = f["X_raman_train"][()]
        y_raman_train = f["y_raman_train"][()]
        X_raman_valid = f["X_raman_test"][()]
        y_raman_valid = f["y_raman_test"][()]
        f.close()

        # Density dataset
        f = h5py.File(path_density, 'r')
        X_density_train = f["X_density_train"][()]
        X_density_valid = f["X_density_valid"][()]
        X_density_test = f["X_density_test"][()]

        y_density_train = f["y_density_train"][()]
        y_density_valid = f["y_density_valid"][()]
        y_density_test = f["y_density_test"][()]
        f.close()

        # Refractive Index (ri) dataset
        f = h5py.File(path_ri, 'r')
        X_ri_train = f["X_ri_train"][()]
        X_ri_valid = f["X_ri_valid"][()]
        X_ri_test = f["X_ri_test"][()]

        lbd_ri_train = f["lbd_ri_train"][()]
        lbd_ri_valid = f["lbd_ri_valid"][()]
        lbd_ri_test = f["lbd_ri_test"][()]

        y_ri_train = f["y_ri_train"][()]
        y_ri_valid = f["y_ri_valid"][()]
        y_ri_test = f["y_ri_test"][()]
        f.close()

        # grabbing number of Raman channels
        self.nb_channels_raman = y_raman_valid.shape[1]

        # preparing data for pytorch

        # Scaler
        # Warning : this was done for tests and currently will not work,
        # as Cp are calculated from unscaled mole fractions...
        if scaling == True:
            X_scaler_mean = np.mean(X_train[:, 0:4], axis=0)
            X_scaler_std = np.std(X_train[:, 0:4], axis=0)
        else:
            X_scaler_mean = 0.0
            X_scaler_std = 1.0

        # The following lines perform scaling (not needed, not active),
        # put the data in torch tensors and send them to device (GPU or CPU, as requested)

        # viscosity
        self.x_visco_train = torch.FloatTensor(self.scaling(X_train[:, 0:4], X_scaler_mean, X_scaler_std)).to(device)
        self.T_visco_train = torch.FloatTensor(X_train[:, 4].reshape(-1, 1)).to(device)
        self.y_visco_train = torch.FloatTensor(y_train[:, 0].reshape(-1, 1)).to(device)

        self.x_visco_valid = torch.FloatTensor(self.scaling(X_valid[:, 0:4], X_scaler_mean, X_scaler_std)).to(device)
        self.T_visco_valid = torch.FloatTensor(X_valid[:, 4].reshape(-1, 1)).to(device)
        self.y_visco_valid = torch.FloatTensor(y_valid[:, 0].reshape(-1, 1)).to(device)

        self.x_visco_test = torch.FloatTensor(self.scaling(X_test[:, 0:4], X_scaler_mean, X_scaler_std)).to(device)
        self.T_visco_test = torch.FloatTensor(X_test[:, 4].reshape(-1, 1)).to(device)
        self.y_visco_test = torch.FloatTensor(y_test[:, 0].reshape(-1, 1)).to(device)

        # entropy
        self.x_entro_train = torch.FloatTensor(self.scaling(X_entropy_train[:, 0:4], X_scaler_mean, X_scaler_std)).to(
            device)
        self.y_entro_train = torch.FloatTensor(y_entropy_train[:, 0].reshape(-1, 1)).to(device)

        self.x_entro_valid = torch.FloatTensor(self.scaling(X_entropy_valid[:, 0:4], X_scaler_mean, X_scaler_std)).to(
            device)
        self.y_entro_valid = torch.FloatTensor(y_entropy_valid[:, 0].reshape(-1, 1)).to(device)

        self.x_entro_test = torch.FloatTensor(self.scaling(X_entropy_test[:, 0:4], X_scaler_mean, X_scaler_std)).to(
            device)
        self.y_entro_test = torch.FloatTensor(y_entropy_test[:, 0].reshape(-1, 1)).to(device)

        # tg
        self.x_tg_train = torch.FloatTensor(self.scaling(X_tg_train[:, 0:4], X_scaler_mean, X_scaler_std)).to(device)
        self.y_tg_train = torch.FloatTensor(y_tg_train.reshape(-1, 1)).to(device)

        self.x_tg_valid = torch.FloatTensor(self.scaling(X_tg_valid[:, 0:4], X_scaler_mean, X_scaler_std)).to(device)
        self.y_tg_valid = torch.FloatTensor(y_tg_valid.reshape(-1, 1)).to(device)

        self.x_tg_test = torch.FloatTensor(self.scaling(X_tg_test[:, 0:4], X_scaler_mean, X_scaler_std)).to(device)
        self.y_tg_test = torch.FloatTensor(y_tg_test.reshape(-1, 1)).to(device)

        # Density
        self.x_density_train = torch.FloatTensor(self.scaling(X_density_train[:, 0:4], X_scaler_mean, X_scaler_std)).to(
            device)
        self.y_density_train = torch.FloatTensor(y_density_train.reshape(-1, 1)).to(device)

        self.x_density_valid = torch.FloatTensor(self.scaling(X_density_valid[:, 0:4], X_scaler_mean, X_scaler_std)).to(
            device)
        self.y_density_valid = torch.FloatTensor(y_density_valid.reshape(-1, 1)).to(device)

        self.x_density_test = torch.FloatTensor(self.scaling(X_density_test[:, 0:4], X_scaler_mean, X_scaler_std)).to(
            device)
        self.y_density_test = torch.FloatTensor(y_density_test.reshape(-1, 1)).to(device)

        # Optical
        self.x_ri_train = torch.FloatTensor(self.scaling(X_ri_train[:, 0:4], X_scaler_mean, X_scaler_std)).to(device)
        self.lbd_ri_train = torch.FloatTensor(lbd_ri_train.reshape(-1, 1)).to(device)
        self.y_ri_train = torch.FloatTensor(y_ri_train.reshape(-1, 1)).to(device)

        self.x_ri_valid = torch.FloatTensor(self.scaling(X_ri_valid[:, 0:4], X_scaler_mean, X_scaler_std)).to(device)
        self.lbd_ri_valid = torch.FloatTensor(lbd_ri_valid.reshape(-1, 1)).to(device)
        self.y_ri_valid = torch.FloatTensor(y_ri_valid.reshape(-1, 1)).to(device)

        self.x_ri_test = torch.FloatTensor(self.scaling(X_ri_test[:, 0:4], X_scaler_mean, X_scaler_std)).to(device)
        self.lbd_ri_test = torch.FloatTensor(lbd_ri_test.reshape(-1, 1)).to(device)
        self.y_ri_test = torch.FloatTensor(y_ri_test.reshape(-1, 1)).to(device)

        # Raman
        self.x_raman_train = torch.FloatTensor(self.scaling(X_raman_train[:, 0:4], X_scaler_mean, X_scaler_std)).to(
            device)
        self.y_raman_train = torch.FloatTensor(y_raman_train).to(device)

        self.x_raman_valid = torch.FloatTensor(self.scaling(X_raman_valid[:, 0:4], X_scaler_mean, X_scaler_std)).to(
            device)
        self.y_raman_valid = torch.FloatTensor(y_raman_valid).to(device)

    def scaling(self, X, mu, s):
        """perform standard scaling"""
        return (X - mu) / s

    def print_data(self):
        """print the specifications of the datasets"""

        print("################################")
        print("#### Dataset specifications ####")
        print("################################")

        # print splitting
        size_train = self.x_visco_train.unique(dim=0).shape[0]
        size_valid = self.x_visco_valid.unique(dim=0).shape[0]
        size_test = self.x_visco_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test

        print("")
        print("Number of unique compositions (viscosity): {}".format(size_total))
        print("Number of unique compositions in training (viscosity): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(size_train / size_total,
                                                                                                size_valid / size_total,
                                                                                                size_test / size_total))

        # print splitting
        size_train = self.x_entro_train.unique(dim=0).shape[0]
        size_valid = self.x_entro_valid.unique(dim=0).shape[0]
        size_test = self.x_entro_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test

        print("")
        print("Number of unique compositions (entropy): {}".format(size_total))
        print("Number of unique compositions in training (entropy): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(size_train / size_total,
                                                                                                size_valid / size_total,
                                                                                                size_test / size_total))

        size_train = self.x_ri_train.unique(dim=0).shape[0]
        size_valid = self.x_ri_valid.unique(dim=0).shape[0]
        size_test = self.x_ri_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test

        print("")
        print("Number of unique compositions (refractive index): {}".format(size_total))
        print("Number of unique compositions in training (refractive index): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(size_train / size_total,
                                                                                                size_valid / size_total,
                                                                                                size_test / size_total))

        size_train = self.x_density_train.unique(dim=0).shape[0]
        size_valid = self.x_density_valid.unique(dim=0).shape[0]
        size_test = self.x_density_test.unique(dim=0).shape[0]
        size_total = size_train + size_valid + size_test

        print("")
        print("Number of unique compositions (density): {}".format(size_total))
        print("Number of unique compositions in training (density): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(size_train / size_total,
                                                                                                size_valid / size_total,
                                                                                                size_test / size_total))

        size_train = self.x_raman_train.unique(dim=0).shape[0]
        size_valid = self.x_raman_valid.unique(dim=0).shape[0]
        size_total = size_train + size_valid

        print("")
        print("Number of unique compositions (Raman): {}".format(size_total))
        print("Number of unique compositions in training (Raman): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid".format(size_train / size_total,
                                                                                size_valid / size_total))

        # training shapes
        print("")
        print("This is for checking the consistency of the dataset...")

        print("Visco train shape")
        print(self.x_visco_train.shape)
        print(self.T_visco_train.shape)
        print(self.y_visco_train.shape)

        print("Entropy train shape")
        print(self.x_entro_train.shape)
        print(self.y_entro_train.shape)

        print("Tg train shape")
        print(self.x_tg_train.shape)
        print(self.y_tg_train.shape)

        print("Density train shape")
        print(self.x_density_train.shape)
        print(self.y_density_train.shape)

        print("Refactive Index train shape")
        print(self.x_ri_train.shape)
        print(self.lbd_ri_train.shape)
        print(self.y_ri_train.shape)

        print("Raman train shape")
        print(self.x_raman_train.shape)
        print(self.y_raman_train.shape)

        # testing device
        print("")
        print("Where are the datasets? CPU or GPU?")

        print("Visco device")
        print(self.x_visco_train.device)
        print(self.T_visco_train.device)
        print(self.y_visco_train.device)

        print("Entropy device")
        print(self.x_entro_train.device)
        print(self.y_entro_train.device)

        print("Tg device")
        print(self.x_tg_train.device)
        print(self.y_tg_train.device)

        print("Density device")
        print(self.x_density_train.device)
        print(self.y_density_train.device)

        print("Refactive Index device")
        print(self.x_ri_test.device)
        print(self.lbd_ri_test.device)
        print(self.y_ri_test.device)

        print("Raman device")
        print(self.x_raman_train.device)
        print(self.y_raman_train.device)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()
print(device)

#
# Loading Data
#

# custom data loader, automatically sent to device
ds = data_loader("./data/NKAS_viscosity_reference.hdf5",
                                  "./data/NKAS_Raman.hdf5",
                                  "./data/NKAS_density.hdf5",
                                  "./data/NKAS_optical.hdf5",
                                  device)


train_melt_X = np.concatenate((ds.x_visco_train.cpu().numpy(), ds.T_visco_train.cpu().numpy()), axis=1)
train_melt_Y = ds.y_visco_train.cpu().numpy()

valid_melt_X = np.concatenate((ds.x_visco_valid.cpu().numpy(), ds.T_visco_valid.cpu().numpy()), axis=1)
valid_melt_Y = ds.y_visco_valid.cpu().numpy()

test_melt_X = np.concatenate((ds.x_visco_test.cpu().numpy(), ds.T_visco_test.cpu().numpy()), axis=1)
test_melt_Y = ds.y_visco_test.cpu().numpy()


A = 1

time1 = time.time()
best_score = 0
for kernel in ["rbf","sigmoid"]:
    for gamma in ["scale",0.001,0.01,0.1,1,10,100]:
        for C in [0.001,0.01,0.1,1,10,100]:
            svm = SVR(gamma=gamma,C=C, kernel=kernel)
            svm.fit(train_melt_X,train_melt_Y)

            # svm.fit(X_train, y_train)
            score = svm.score(valid_melt_X,valid_melt_Y)
            if score > best_score:
                best_score = score
                best_parameters = {'gamma':gamma,'C':C,"kernel":kernel}
svm = SVR(**best_parameters) #使用最佳参数，构建新的模型
svm.fit(train_melt_X,train_melt_Y) #使用训练集和验证集进行训练，more data always results in good performance.
test_score = svm.score(test_melt_X,test_melt_Y) # evaluation模型评估

predicty1 = svm.predict(train_melt_X)
result1=mean_squared_error(train_melt_Y, predicty1 )
predicty2 = svm.predict(valid_melt_X)
result2=mean_squared_error(valid_melt_Y, predicty2 )
time2 = time.time()
predicty3 = svm.predict(test_melt_X)
result3=mean_squared_error(test_melt_Y, predicty3 )
resultn = abs(test_melt_Y.reshape(-1,1)-predicty3.reshape(-1,1))
print("Best score on validation set:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
print("Best score on test set:{:.2f}".format(test_score))
print("Best mse on train set:{:.2f}".format(result1))
print("Best mse on valid set:{:.2f}".format(result2))
print("Best mse on test set:{:.2f}".format(result3))
time3 = time.time()

print("train:"+str(time2-time1))
print("prediction:"+str(time3-time2))
np.savetxt("./each_point/svr/svr_point_error_"+str(seed-162)+".csv",resultn,delimiter=",")
# best_score = 0.0
# for gamma in [0.001,0.01,0.1,1,10,100]:
#     for C in [0.001,0.01,0.1,1,10,100]:
#         svm = SVR(gamma=gamma,C=C)
#         scores = cross_val_score(svm,X_trainval,y_trainval,cv=5) #5折交叉验证
#         score = scores.mean() #取平均数
#         if score > best_score:
#             best_score = score
#             best_parameters = {"gamma":gamma,"C":C}
# svm = SVR(**best_parameters)
# svm.fit(X_trainval,y_trainval)
# test_score = svm.score(X_test,y_test)
# print("Best score on validation set:{:.2f}".format(best_score))
# print("Best parameters:{}".format(best_parameters))
# print("Score on testing set:{:.2f}".format(test_score))
