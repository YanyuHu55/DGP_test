import json
import numpy as np
import os
import torch
import tqdm
import math
import numpy as np
import gpytorch
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import random


from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import VariationalStrategy,MeanFieldVariationalDistribution,CholeskyVariationalDistribution,LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.models.exact_gp import GP
from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from matplotlib import pyplot as plt
from tqdm.notebook import trange
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from matplotlib import pyplot as plt
import argparse

# current_device = torch.cuda.current_device()
# torch.cuda.set_device(current_device)

params = argparse.ArgumentParser()
params.add_argument('-num_search', type=int, default=10, help='iteration time of random searching ')
params.add_argument('-repeat_time', type=int, default=1, help='repeat time of random searching ')
params.add_argument('-cross_split', type=int, default=2, help='number of cross validation split ')
params.add_argument('-params_epoch', type=int, default=10, help=' epoch number of finding parameters ')
params.add_argument('-optimal_epoch', type=int, default=10, help=' epoch number of optimal parameters ')
# params.add_argument('-num_hidden_dgp_dims', type=int, default=1, help=' the number of hidden layer dimension ')
params.add_argument('-optimizer_lr', type=float, default=0.01, help=' the learning rate of the optimizer ')

args = params.parse_args()

num_search = args.num_search
repeat_time = args.repeat_time
cross_split = args.cross_split
params_epoch = args.params_epoch
optimal_epoch = args.optimal_epoch
# num_hidden_dgp_dims = args.num_hidden_dgp_dims
optimizer_lr = args.optimizer_lr



torch.manual_seed(8)
smoke_test = ('CI' in os.environ)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_file = 'april.json'
with open(data_file) as f:
    data = json.load(f)
collect_data = {}
receiver_position = torch.empty(0, dtype=torch.float)
# print(receiver_position)
RSS_value = torch.empty(0)
# print(RSS_value)
for key, value in data.items():
    rx_data = value['rx_data']
    # print(rx_data)
    metadata = value['metadata']
    # print(metadata)
    power = metadata[0]['power']
    # print(power)
    base_station = rx_data[3][3]
    # print(base_station)
    tr_cords = torch.tensor(value["tx_coords"]).float()
    # print(tr_cords)
    if base_station == 'cbrssdr1-honors-comp' and power == 1: # 'cbrssdr1-bes-comp', 'cbrssdr1-honors-comp'
        RSS_sample = torch.tensor([rx_data[3][0]]).float()
        # print(RSS_sample)
        RSS_value = torch.cat((RSS_value, RSS_sample), dim=0)
        # print(RSS)
        # print(RSS.shape)
        receiver_position = torch.cat((receiver_position, tr_cords), dim=0)
        # print(location)
        # print(location.shape)
# print(RSS_value)
# print(RSS_value.shape) # torch.Size([326])
# print(receiver_position)
# print(receiver_position.shape) # torch.Size([326, 2])
# process RSS_value and receiver_position data
RSS_value = RSS_value.view(RSS_value.size(0),1)
for i in range(len(receiver_position)):
    receiver_position[i][0] = (receiver_position[i][0] - 40.75) * 1000
    receiver_position[i][1] = (receiver_position[i][1] + 111.83) * 1000
# print('original RSS value',RSS_value)
# print(RSS_value.shape)

shuffle_index = torch.randperm(len(receiver_position))
receiver_position = receiver_position[shuffle_index].to(device)
RSS_value = RSS_value[shuffle_index].to(device)
print('RSS value', RSS_value)

test_x = receiver_position[176:,:] # cbrssdr1-honors-comp: 409, 359, 309, 189, (296)
test_y = RSS_value[176:,:]
train_x = receiver_position[0:176,:]
train_y = RSS_value[0:176,:]
# print('decimal train x',train_x)

# normalize the train x and test x
mean_norm_x, std_norm_x = train_x.mean(dim=0),train_x.std(dim=0)
train_x = (train_x - mean_norm_x) / (std_norm_x)
test_x = (test_x - mean_norm_x) / (std_norm_x)

# normalize the train y and test y
mean_norm, std_norm = train_y.mean(dim=0),train_y.std(dim=0)
train_y = (train_y - mean_norm) / (std_norm)
test_y = (test_y - mean_norm) / (std_norm)

train_y = 10 ** (train_y / 20)*100000
test_y = 10 ** (test_y / 20)*100000

mean_norm_decimal, std_norm_decimal = train_y.mean(dim=0),train_y.std(dim=0)
train_y = (train_y - mean_norm_decimal) / (std_norm_decimal)
test_y = (test_y - mean_norm_decimal) / (std_norm_decimal)

train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

print(RSS_value.shape) # 326, 485
print('decimal train x',train_x.shape)
# print('decimal train y',train_y)


def initialize_inducing_inputs(X, M):
    kmeans = KMeans(n_clusters=M)
    # print('kmeans',kmeans)
    kmeans.fit(X.cpu())
    # print('kmeans.fit(X)', kmeans.fit(X))
    inducing_inputs = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    return inducing_inputs
def _init_pca(X, latent_dim):
    U, S, V = torch.pca_lowrank(X, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(X, V[:,:latent_dim]))
def increase_dim(X, latent_dim):
    X = X.cpu().numpy()
    n_samples, n_features = X.shape
    features = [X[:, i] for i in range(n_features)]
    # print('X[:, 0]',X[:, 0])
    # print('X[:, 1]', X[:, 1])
    # print('feature:',features)
    for i in range(2, latent_dim):
        new_feature = (X[:, 0]+X[:, 1]) ** i
        features.append(new_feature)
    X_expanded = np.column_stack(features)
    inducing_inputs=torch.tensor(X_expanded, dtype=torch.float32).to(device)
    return inducing_inputs

# Deep Gaussian Process
class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, linear_mean = False): # 40, 70,100, 160;
        # inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        if input_dims == train_x.shape[-1]:
            inducing_points = torch.empty(0, dtype=torch.float32).to(device)

            for i in range(output_dims):
                inducing_points_i = initialize_inducing_inputs(train_x, num_inducing)
                inducing_points_i = torch.unsqueeze(inducing_points_i, 0)
                # print(inducing_points_i)
                inducing_points = torch.cat((inducing_points, inducing_points_i)).to(torch.float32)
                # print('inducing points for 2', inducing_points)
                # print('inducing point shape2', inducing_points.shape)
        elif input_dims > train_x.shape[-1]:
            inducing_points = torch.empty(0, dtype=torch.float32).to(device)
            for i in range(output_dims):
                inducing_points_i = initialize_inducing_inputs(increase_dim(train_x, input_dims).detach(), num_inducing)
                inducing_points_i = torch.unsqueeze(inducing_points_i, 0)
                # print(inducing_points_i)
                inducing_points = torch.cat((inducing_points, inducing_points_i)).to(torch.float32)
                # print('inducing points for m', inducing_points)
                # print('inducing point shapem', inducing_points.shape)
        else:
            inducing_points = torch.empty(0, dtype=torch.float32).to(device)
            for i in range(output_dims):
                inducing_points_i = initialize_inducing_inputs(_init_pca(train_x, input_dims).detach(), num_inducing)
                inducing_points_i = torch.unsqueeze(inducing_points_i, 0)
                # print(inducing_points_i)
                inducing_points = torch.cat((inducing_points, inducing_points_i)).to(torch.float32)
                # print('inducing points for 2', inducing_points)
        print('inducing points shape', inducing_points.shape)
        batch_shape = torch.Size([output_dims])
        # mean_field variational distribution
        # variational_distribution = MeanFieldVariationalDistribution(
        #     num_inducing_points=num_inducing,
        #     batch_shape=batch_shape
        # )
        # variational_distribution = MeanFieldVariationalDistribution.initialize_variational_distribution()

        # print('variational variational_mean',variational_distribution.variational_mean)
        # print('variational variational_stddev', variational_distribution.variational_stddev)
        # print(variational_distribution.covariance_matrix)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        # include the inducing points
        mean_x =self.mean_module(x)
        # print('mean_x',mean_x)
        covar_x = self.covar_module(x)
        # print('covar_x',covar_x)
        return MultivariateNormal(mean_x, covar_x)

num_tasks = train_y.size(-1)
# num_hidden_dgp_dims = 1
num_hidden_dims_choice = [1]
hiden_lengthscale_choice = [0.1]
# hiden_lengthscale_choice_2 = [0.25]
hiden_outputscale_choise = [0.1] # [-3.0, -2.0, -1.0] [-0.1, -0.01, 0.1]
last_lengthscale_choice  = [0.1]# last_lengthscale_choice  = [-0.1, 0.1, 0.01, -0.01]
last_outputscale_choise  = [0.1] # -1.5, -1.75, 0.01
likelihood_noise_range   = [0.01]
# num_inducing_choice = [i for i in range(math.floor(train_x.size(0) / 3),train_x.size(0)-1, 5)] # # time:40
# num_inducing_choice = [i for i in range(173,train_x.size(0)-1, 5)]
# num_inducing_choice = [i for i in range(3, math.floor(train_x.size(0) / 3), 5)] # times 19

num_inducing_choice = [123]

best_train = float('inf')
# best_num_inducing = 1
second_best_train = float('inf')
third_best_train = float('inf')
fourth_best_train = float('inf')
fifth_best_train = float('inf')
sixth_best_train = float('inf')
seventh_best_train = float('inf')
eighth_best_train = float('inf')
ninth_best_train = float('inf')
tenth_best_train = float('inf')

best_hyperparams = None
best_result = None
second_best_hyperparams = None
second_best_result = None
third_best_hyperparams = None
third_best_result = None
fourth_best_hyperparams = None
fourth_best_result = None
fifth_best_hyperparams = None
fifth_best_result = None
sixth_best_hyperparams = None
sixth_best_result = None
seventh_best_hyperparams = None
seventh_best_result = None
eighth_best_hyperparams = None
eighth_best_result = None
ninth_best_hyperparams = None
ninth_best_result = None
tenth_best_hyperparams = None
tenth_best_result = None

worst_train = 0
worst_hyperparams = None
worst_result = None

# random searching and separate validation--data2

# grid searching and separate validation--data2
method = ['grid searching and separate validation--data2']
grid_times = 0

for likelihood_noise_value in likelihood_noise_range:
    for hidden_layer_outputscale in hiden_outputscale_choise:
        for last_layer_lengthscale in last_lengthscale_choice:
            for last_layer_outputscale in last_outputscale_choise:
                for hidden_layer_lengthscale_1 in hiden_lengthscale_choice:
                    for num_hidden_dgp_dims in num_hidden_dims_choice:
                        for num_inducing in num_inducing_choice:
                            hidden_layer_lengthscale_2 = hidden_layer_lengthscale_1
                            total_train_rmse = 0.0
                            total_test_rmse = 0.0
                            total_train_rmse_dB = 0.0
                            total_test_rmse_dB = 0.0
                            total_loss = 0.0
                            grid_times = grid_times + 1
                            print(grid_times,
                                  '. hidden_lengthscale: ',
                                  [hidden_layer_lengthscale_1, hidden_layer_lengthscale_2],
                                  ';\n   hidden_outputscale: ', hidden_layer_outputscale,
                                  ';\n   last_lengthscale: ',
                                  [last_layer_lengthscale for i in range(num_hidden_dgp_dims)],
                                  ';\n   last_outputscale: ', last_layer_outputscale,
                                  ';\n   noise: ', likelihood_noise_value,
                                  ';\n   number of hidden dim: ', num_hidden_dgp_dims,
                                  ';\n   num of inducing: ', num_inducing,
                                  )
                            for time in range(repeat_time):
                                class MultitaskDeepGP(DeepGP):
                                    def __init__(self, train_x_shape):
                                        hidden_layer = DGPHiddenLayer(
                                            input_dims=train_x_shape[-1],
                                            output_dims=num_hidden_dgp_dims,
                                            num_inducing=num_inducing,
                                            linear_mean=True
                                        )

                                        # second_hidden_layer = DGPHiddenLayer(
                                        #     input_dims=hidden_layer.output_dims,
                                        #     output_dims=num_hidden_dgp_dims+1,
                                        #     linear_mean=True
                                        # )
                                        #
                                        # third_hidden_layer = DGPHiddenLayer(
                                        #     input_dims=second_hidden_layer.output_dims+train_x_shape[-1],
                                        #     output_dims=num_hidden_dgp_dims+2,
                                        #     linear_mean=True
                                        # )

                                        last_layer = DGPHiddenLayer(
                                            input_dims=hidden_layer.output_dims,
                                            output_dims=num_tasks,
                                            num_inducing=num_inducing,
                                            linear_mean=True
                                        )
                                        super().__init__()

                                        self.hidden_layer = hidden_layer
                                        # self.second_hidden_layer = second_hidden_layer
                                        # self.third_hidden_layer = third_hidden_layer
                                        self.last_layer = last_layer

                                        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

                                    def forward(self, inputs, **kwargs):
                                        hidden_rep1 = self.hidden_layer(inputs)
                                        # print('22',inputs.shape)
                                        # print('11',hidden_rep1)
                                        # hidden_rep2 = self.second_hidden_layer(hidden_rep1, **kwargs)
                                        # hidden_rep3 = self.third_hidden_layer(hidden_rep2, inputs, **kwargs)
                                        output = self.last_layer(hidden_rep1)
                                        return output

                                    def predict(self, test_x):
                                        with torch.no_grad():
                                            preds = model.likelihood(model(test_x)).to_data_independent_dist()
                                        return preds.mean.mean(0), preds.variance.mean(0)


                                model = MultitaskDeepGP(train_x.shape)
                                if torch.cuda.is_available():
                                    model = model.cuda()

                                # hypers = {
                                #     'hidden_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor([[[-3.0, -2.5]]]).to(device), #-3,
                                #     'hidden_layer.covar_module.raw_outputscale': torch.tensor([-1.0]).to(device),
                                #     'last_layer.covar_module.base_kernel.raw_lengthscale':torch.tensor([[[6.0]]]).to(device),
                                #     'last_layer.covar_module.raw_outputscale': torch.tensor([-0.1]).to(device),  # 0, 1, 0.5
                                #     'likelihood.raw_task_noises':torch.tensor([0.01]).to(device),
                                #     'likelihood.raw_noise':torch.tensor([0.01]).to(device),
                                # }

                                hypers = {
                                    'hidden_layer.covar_module.base_kernel.lengthscale': torch.tensor(
                                        [[[hidden_layer_lengthscale_1, hidden_layer_lengthscale_2]]]).to(device),
                                    # -3,
                                    'hidden_layer.covar_module.outputscale': torch.tensor(
                                        [hidden_layer_outputscale]).to(device),
                                    'last_layer.covar_module.base_kernel.lengthscale': torch.tensor(
                                        [[[last_layer_lengthscale for i in range(num_hidden_dgp_dims)]]]).to(
                                        device),
                                    'last_layer.covar_module.outputscale': torch.tensor(
                                        [last_layer_outputscale]).to(device),  # 0, 1, 0.5
                                    'likelihood.task_noises': torch.tensor([likelihood_noise_value]).to(device),
                                    'likelihood.noise': torch.tensor([likelihood_noise_value]).to(device),
                                }

                                model.initialize(**hypers)
                                model.train()
                                optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
                                # optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_lr)
                                mll = DeepApproximateMLL(
                                    VariationalELBO(model.likelihood, model, num_data=train_y.size(0))).to(device)
                                # num_epochs = 1 if smoke_test else 100
                                # num_epochs = 1 if smoke_test else 2000 best
                                epochs_iter = tqdm.tqdm(range(params_epoch), desc='Epoch')
                                loss_set = np.array([])
                                for i in epochs_iter:
                                    optimizer.zero_grad()
                                    output = model(train_x)
                                    loss = -mll(output, train_y)
                                    epochs_iter.set_postfix(loss='{:0.5f}'.format(loss.item()))
                                    loss_set = np.append(loss_set, loss.item())
                                    loss.backward()
                                    optimizer.step()
                                total_loss += loss.item()
                                model.eval()
                                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                                    predictions, predictive_variance = model.predict(test_x.float())
                                predictions = predictions * std_norm_decimal + mean_norm_decimal
                                test_y = test_y * std_norm_decimal + mean_norm_decimal
                                predictions = 20 * torch.log10(predictions / 100000)
                                test_y = 20 * torch.log10(test_y / 100000)

                                predictions = predictions * std_norm + mean_norm
                                test_y = test_y * std_norm + mean_norm

                                predictions = 10 ** (predictions / 20)
                                test_y = 10 ** (test_y / 20)

                                # print('noise_covar',noise_convar)
                                # print(predictive_variance)
                                for task in range(0, 1):
                                    # print(task)
                                    # print(predictions[:, task])

                                    test_rmse = torch.mean(
                                        torch.pow(predictions[:, task] - test_y[:, task], 2)).sqrt()
                                    total_test_rmse += test_rmse.item()
                                    test_y = 20 * torch.log10(test_y)
                                    test_rmse_dB = torch.mean(
                                        torch.pow(predictions[:, task] - test_y[:, task], 2)).sqrt()
                                    total_test_rmse_dB += test_rmse_dB.item()
                                model.eval()
                                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                                    predictions, predictive_variance = model.predict(train_x.float())
                                predictions = predictions * std_norm_decimal + mean_norm_decimal
                                train_y = train_y * std_norm_decimal + mean_norm_decimal
                                predictions = 20 * torch.log10(predictions / 100000)
                                train_y = 20 * torch.log10(train_y / 100000)

                                predictions = predictions * std_norm + mean_norm
                                train_y = train_y * std_norm + mean_norm

                                predictions = 10 ** (predictions / 20)
                                train_y = 10 ** (train_y / 20)
                                for task in range(0, 1):
                                    # print(task)
                                    # print(predictions[:, task])
                                    train_rmse = torch.mean(
                                        torch.pow(predictions[:, task] - train_y[:, task], 2)).sqrt()
                                    total_train_rmse += train_rmse.item()
                                    train_y = 20 * torch.log10(train_y)
                                    train_rmse_dB = torch.mean(
                                        torch.pow(predictions[:, task] - train_y[:, task], 2)).sqrt()
                                    total_train_rmse_dB += train_rmse_dB.item()

                                print('    loss:', loss.item(),
                                      ': test rmse in decimal: %e' % test_rmse,
                                      ': test rmse in dB:%e' % test_rmse_dB,
                                      '; \033[1mtrain rmse in decimal: %e' % train_rmse, '\033[0m',
                                      '; \033[1mtrain rmse in dB: %e' % train_rmse_dB, '\033[0m'
                                      )
                                # normalize the train y and test y
                                train_y = (train_y - mean_norm) / (std_norm)
                                test_y = (test_y - mean_norm) / (std_norm)

                                train_y = 10 ** (train_y / 20) * 100000
                                test_y = 10 ** (test_y / 20) * 100000

                                train_y = (train_y - mean_norm_decimal) / (std_norm_decimal)
                                test_y = (test_y - mean_norm_decimal) / (std_norm_decimal)
                            print('\033[1m', grid_times, '\033[0m',
                                  '. hidden_lengthscale: ',
                                  [hidden_layer_lengthscale_1, hidden_layer_lengthscale_2],
                                  ';\n   hidden_outputscale: ', hidden_layer_outputscale,
                                  ';\n   last_lengthscale: ',
                                  [last_layer_lengthscale for i in range(num_hidden_dgp_dims)],
                                  ';\n   last_outputscale: ', last_layer_outputscale,
                                  ';\n   noise: ', likelihood_noise_value,
                                  ';\n   number of hidden dim: ', num_hidden_dgp_dims,
                                  ';\n   number of inducing: ', num_inducing,
                                  )
                            print('    loss:', total_loss / repeat_time,
                                  ': test rmse in decimal: %e' % (total_test_rmse / repeat_time),
                                  ': test rmse in dB: %e' % (total_test_rmse_dB / repeat_time),
                                  '; \033[1mtrain rmse in decmial: %e '% (total_train_rmse / repeat_time), '\033[0m',
                                  '; \033[1mtrain rmse in dB: %e ' % (total_train_rmse_dB / repeat_time), '\033[0m'
                                  )
                            if (total_train_rmse / repeat_time) < best_train:
                                # best_num_inducing = num_inducing
                                tenth_best_train = ninth_best_train
                                ninth_best_train = eighth_best_train
                                eighth_best_train = seventh_best_train
                                seventh_best_train = sixth_best_train
                                sixth_best_train = fifth_best_train
                                fifth_best_train = fourth_best_train
                                fourth_best_train = third_best_train
                                third_best_train = second_best_train
                                second_best_train = best_train
                                best_train = total_train_rmse / repeat_time

                                tenth_best_hyperparams = ninth_best_hyperparams
                                ninth_best_hyperparams = eighth_best_hyperparams
                                eighth_best_hyperparams = seventh_best_hyperparams
                                seventh_best_hyperparams = sixth_best_hyperparams
                                sixth_best_hyperparams = fifth_best_hyperparams
                                fifth_best_hyperparams = fourth_best_hyperparams
                                fourth_best_hyperparams = third_best_hyperparams
                                third_best_hyperparams = second_best_hyperparams
                                second_best_hyperparams = best_hyperparams
                                best_hyperparams = {
                                    'hidden_layer_lengthscale': [hidden_layer_lengthscale_1,
                                                                 hidden_layer_lengthscale_2],
                                    'hidden_layer_outputscale': hidden_layer_outputscale,
                                    'last_layer_lengthscale': [last_layer_lengthscale for i in
                                                               range(num_hidden_dgp_dims)],
                                    'last_layer_outputscale': last_layer_outputscale,
                                    'likelihood_noise': likelihood_noise_value,
                                    'num_hidden_dim': num_hidden_dgp_dims,
                                    'num_inducing': num_inducing
                                }

                                tenth_best_result = ninth_best_result
                                ninth_best_result = eighth_best_result
                                eighth_best_result = seventh_best_result
                                seventh_best_result = sixth_best_result
                                sixth_best_result = fifth_best_result
                                fifth_best_result = fourth_best_result
                                fourth_best_result = third_best_result
                                third_best_result = second_best_result
                                second_best_result = best_result
                                best_result = {
                                    'loss': total_loss / repeat_time,
                                    'test rmse': total_test_rmse / repeat_time,
                                    'train rmse': total_train_rmse / repeat_time
                                }
                            if (total_train_rmse / repeat_time) > worst_train:
                                worst_train = total_train_rmse / repeat_time
                                worst_hyperparams = {
                                    'hidden_layer_lengthscale': [hidden_layer_lengthscale_1,
                                                                 hidden_layer_lengthscale_2],
                                    'hidden_layer_outputscale': hidden_layer_outputscale,
                                    'last_layer_lengthscale': [last_layer_lengthscale for i in
                                                               range(num_hidden_dgp_dims)],
                                    'last_layer_outputscale': last_layer_outputscale,
                                    'likelihood_noise': likelihood_noise_value,
                                    'num_hidden_dim': num_hidden_dgp_dims,
                                    'num_inducing': num_inducing
                                }
                                worst_result = {
                                    'loss': total_loss / repeat_time,
                                    'test rmse': total_test_rmse / repeat_time,
                                    'train rmse': total_train_rmse / repeat_time
                                }

num_inducing = best_hyperparams['num_inducing']
num_hidden_dgp_dims = best_hyperparams['num_hidden_dim']

class MultitaskDeepGP(DeepGP):
    def __init__(self,train_x_shape):
        hidden_layer = DGPHiddenLayer(
            input_dims = train_x_shape[-1],
            output_dims= num_hidden_dgp_dims,
            num_inducing=num_inducing,
            linear_mean=True
        )

        # second_hidden_layer = DGPHiddenLayer(
        #     input_dims=hidden_layer.output_dims,
        #     output_dims=num_hidden_dgp_dims+1,
        #     linear_mean=True
        # )
        #
        # third_hidden_layer = DGPHiddenLayer(
        #     input_dims=second_hidden_layer.output_dims+train_x_shape[-1],
        #     output_dims=num_hidden_dgp_dims+2,
        #     linear_mean=True
        # )

        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            num_inducing=num_inducing,
            linear_mean=True
        )
        super().__init__()

        self.hidden_layer = hidden_layer
        # self.second_hidden_layer = second_hidden_layer
        # self.third_hidden_layer = third_hidden_layer
        self.last_layer = last_layer

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs,**kwargs):

        hidden_rep1 = self.hidden_layer(inputs)
        # print('22',inputs.shape)
        # print('11',hidden_rep1)
        # hidden_rep2 = self.second_hidden_layer(hidden_rep1, **kwargs)
        # hidden_rep3 = self.third_hidden_layer(hidden_rep2, inputs, **kwargs)
        output = self.last_layer(hidden_rep1)
        return output
    def predict(self, test_x):
        with torch.no_grad():
            preds = model.likelihood(model(test_x)).to_data_independent_dist()
        return preds.mean.mean(0), preds.variance.mean(0)

model = MultitaskDeepGP(train_x.shape)
if torch.cuda.is_available():
    model = model.cuda()


# hidden layer


# hypers = {
#     'hidden_layer.covar_module.base_kernel.raw_lengthscale': torch.tensor([[[-3.0, -2.5]]]).to(device), #-3,
#     'hidden_layer.covar_module.raw_outputscale': torch.tensor([-1.0]).to(device),
#     'last_layer.covar_module.base_kernel.raw_lengthscale':torch.tensor([[[6.0]]]).to(device),
#     'last_layer.covar_module.raw_outputscale': torch.tensor([-0.1]).to(device),  # 0, 1, 0.5
#     'likelihood.raw_task_noises':torch.tensor([0.01]).to(device),
#     'likelihood.raw_noise':torch.tensor([0.01]).to(device),
# }

hypers = {
    'hidden_layer.covar_module.base_kernel.lengthscale': torch.tensor([[best_hyperparams['hidden_layer_lengthscale']]]).to(device), #-3,
    'hidden_layer.covar_module.outputscale': torch.tensor([best_hyperparams['hidden_layer_outputscale']]).to(device),
    'last_layer.covar_module.base_kernel.lengthscale':torch.tensor([[best_hyperparams['last_layer_lengthscale']]]).to(device),
    'last_layer.covar_module.outputscale': torch.tensor([best_hyperparams['last_layer_outputscale']]).to(device),  # 0, 1, 0.5
    'likelihood.task_noises':torch.tensor([best_hyperparams['likelihood_noise']]).to(device),
    'likelihood.noise':torch.tensor([best_hyperparams['likelihood_noise']]).to(device),
}


model.initialize(**hypers)
# model likelihood
print('\033[1mBefore training\033[0m')
print(f'Actual hidden lengthscale: {model.hidden_layer.covar_module.base_kernel.lengthscale}')
print(f'raw hidden lengthscale: {model.hidden_layer.covar_module.base_kernel.raw_lengthscale}')
print(f'Actual hidden outputscale: {model.hidden_layer.covar_module.outputscale}')
print(f'raw hidden outputscale: {model.hidden_layer.covar_module.raw_outputscale} ')

print(f'Actual last lengthscale: {model.last_layer.covar_module.base_kernel.lengthscale}')
print(f'raw last lengthscale: {model.last_layer.covar_module.base_kernel.raw_lengthscale}')
print(f'Actual last outputscale: {model.last_layer.covar_module.outputscale}')
print(f'raw last outputscale: {model.last_layer.covar_module.raw_outputscale} ')
print(f'Actual likelihood noise: {model.likelihood.noise}')
print(f'raw likelihood noise: {model.likelihood.raw_noise} ')




# model.initialize(**hypers)
# print('hidden_layer.variational_strategy.inducing_points ',model.hidden_layer.variational_strategy.inducing_points)
# print('last_layer.variational_strategy.inducing_points ',model.last_layer.variational_strategy.inducing_points)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_lr)

mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model,num_data=train_y.size(0))).to(device)
num_epochs = 1 if smoke_test else optimal_epoch
# num_epochs = 1 if smoke_test else 2000 best
epochs_iter = tqdm.tqdm(range(num_epochs),desc='Epoch')
loss_set = np.array([])
for i in epochs_iter:
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    epochs_iter.set_postfix(loss='{:0.5f}'.format(loss.item()))
    loss_set = np.append(loss_set, loss.item())
    loss.backward()
    optimizer.step()


model.eval()
for param_name, param in model.named_parameters():

    if param_name == 'likelihood.raw_noise':
        noise_convar = param.item()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions, predictive_variance = model.predict(test_x.float())
    lower = predictions - 1.96 * predictive_variance.sqrt()
    upper = predictions + 1.96 * predictive_variance.sqrt()

print('\033[1mAfter training\033[0m')
print(f'Actual hidden lengthscale (after): {model.hidden_layer.covar_module.base_kernel.lengthscale}')
print(f'raw hidden lengthscale (after): {model.hidden_layer.covar_module.base_kernel.raw_lengthscale}')
print(f'Actual hidden outputscale (after): {model.hidden_layer.covar_module.outputscale}')
print(f'raw hidden outputscale (after): {model.hidden_layer.covar_module.raw_outputscale} ')

print(f'Actual last lengthscale (after): {model.last_layer.covar_module.base_kernel.lengthscale}')
print(f'raw last lengthscale (after): {model.last_layer.covar_module.base_kernel.raw_lengthscale}')
print(f'Actual last outputscale (after): {model.last_layer.covar_module.outputscale}')
print(f'raw last outputscale (after): {model.last_layer.covar_module.raw_outputscale} ')
print(f'Actual likelihood noise (after): {model.likelihood.noise}')
print(f'raw likelihood noise (after): {model.likelihood.raw_noise} ')

# print(predictions)
# print(predictions.size)
predictions = predictions*std_norm_decimal +mean_norm_decimal
test_y = test_y *std_norm_decimal + mean_norm_decimal
predictions = 20 * torch.log10(predictions/100000)
test_y = 20 * torch.log10(test_y/100000)

# print('std_norm_decimal',std_norm_decimal)
# print('mean_norm_decimal',mean_norm_decimal)
# print('std_norm',std_norm)
# print('mean_norm',mean_norm)

# lower and upper
lower = lower*std_norm_decimal +mean_norm_decimal
lower = 20 * torch.log10(lower/100000)
lower = lower*std_norm +mean_norm
upper = upper*std_norm_decimal +mean_norm_decimal
upper = 20 * torch.log10(upper/100000)
upper = upper*std_norm +mean_norm
# print('95% lower confident interval', lower)
# print('95% upper confident interval', upper)

# covar = covar *std_norm
# noise_convar = noise_convar*std_norm
predictions = predictions*std_norm +mean_norm
test_y = test_y *std_norm + mean_norm
print('predicted test y',predictions[:].view(1,test_x.size(0)))
print('original test y',test_y[:].view(1,test_x.size(0)))
predictions = 10 ** (predictions / 20)
test_y = 10 ** (test_y / 20)
# print('noise_covar',noise_convar)
# print(predictive_variance)


# test confident region
fig, ax = plt.subplots(figsize=(18, 4))
x = np.arange(1, test_x.size(0)+1)
# print('x', x)
y = predictions[:].view(1,test_y.size(0)).squeeze().detach()
# print('y',y)
y = y.cpu().numpy()
# print('y',y)
ax.plot(x, y, 'b')
ax.plot(x, test_y.cpu().view(1,test_x.size(0)).squeeze().detach().numpy(), 'k*')
ax.fill_between(x, lower.cpu().view(1,test_x.size(0)).squeeze().detach().numpy(), upper.cpu().view(1,test_x.size(0)).squeeze().detach().numpy(), alpha=0.5)
ax.set_ylabel('RSS value')
ax.set_xlabel('data point')
ax.set_ylim([-105, -50])
ax.legend(['Mean','Observed Data',  'Confidence'], fontsize='x-small')
ax.set_title('DSVI DGP test confident interval-train %d point(s)/ test %d point(s)' % (len(train_y),len(test_y)))
plt.show()


for task in range(0,1):
    # print(task)
    # print(predictions[:, task])

    rmse = torch.mean(torch.pow(predictions[:, task] - test_y[:, task], 2)).sqrt()
    print(task + 1, '. \033[1mtest RMSE in decimal:%e'% rmse,'\033[0m')
    test_y = 20 * torch.log10(test_y)
    rmse = torch.mean(torch.pow(predictions[:, task] - test_y[:, task], 2)).sqrt()
    print(task + 1, '. \033[1mtest RMSE in dB: %e' % rmse, '\033[0m')




# training error
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions, predictive_variance = model.predict(train_x.float())
    # mean = predictions.mean
    # lower, upper = predictions.confidence_region()
    lower = predictions - 1.96 * predictive_variance.sqrt()
    upper = predictions + 1.96 * predictive_variance.sqrt()
# print(predictions)
# print(predictions.size)
# print('noise_covar',noise_convar)
print('One transmitter(y1)')
print('For Gaussian Process, 31 train / 2 test')
# print('noise_covar',noise_convar)
print('test error')
# print('noise_covar',noise_convar)
# print(predictive_variance)
print('final loss',loss.item())
# print('iteration num', iteration_num)
predictions = predictions*std_norm_decimal +mean_norm_decimal
train_y = train_y *std_norm_decimal + mean_norm_decimal
predictions = 20 * torch.log10(predictions/100000)
train_y = 20 * torch.log10(train_y/100000)
# covar = covar *std_norm
# noise_convar = noise_convar*std_norm
predictions = predictions*std_norm +mean_norm
train_y = train_y *std_norm + mean_norm
print('predicted train y',predictions[:].view(1,train_x.size(0)))
print('original train data',train_y[:].view(1,train_x.size(0)))
predictions = 10 ** (predictions / 20)
train_y = 10 ** (train_y / 20)


# lower and upper
# print('std_norm_decimal',std_norm_decimal)
# print('mean_norm_decimal',mean_norm_decimal)
# print('std_norm',std_norm)
# print('mean_norm',mean_norm)
lower = lower*std_norm_decimal +mean_norm_decimal
lower = 20 * torch.log10(lower/100000)
lower = lower*std_norm +mean_norm
upper = upper*std_norm_decimal +mean_norm_decimal
upper = 20 * torch.log10(upper/100000)
upper = upper*std_norm +mean_norm
# print('95% lower confident interval', lower)
# print('95% upper confident interval', upper)

for task in range(0,1):
    # print(task)
    # print(predictions[:, task])
    rmse = torch.mean(torch.pow(predictions[:, task] - train_y[:, task], 2)).sqrt()
    print(task + 1, '. \033[1mtrain RMSE in decimal: %e' % rmse,'\033[0m')
    train_y = 20 * torch.log10(train_y)
    rmse = torch.mean(torch.pow(predictions[:, task] - train_y[:, task], 2)).sqrt()
    print(task + 1, '. \033[1mtrain RMSE in dB: %e' % rmse, '\033[0m')


# train confident region
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(1, train_x.size(0)+1)
# print('x', x)
y = predictions[:].view(1,train_y.size(0)).squeeze().detach()
# print('y',y)
y = y.cpu().numpy()
# print('y',y)
ax.plot(x, y, 'b')
ax.plot(x, train_y.cpu().view(1,train_x.size(0)).squeeze().detach().numpy(), 'k*')
ax.fill_between(x, lower.cpu().view(1,train_x.size(0)).squeeze().detach().numpy(), upper.cpu().view(1,train_x.size(0)).squeeze().detach().numpy(), alpha=0.5)
ax.set_ylabel('RSS value')
ax.set_xlabel('data point')
ax.set_ylim([-105, -50])
ax.legend(['Mean','Observed Data',  'Confidence'], fontsize='x-small')
ax.set_title('DSVI DGP train confident interval-train %d point(s)/ test %d point(s)' % (len(train_y),len(test_y)))
plt.show()


# loss figure
x = np.arange(1, num_epochs+1)
y = loss_set
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, linewidth=1.0)
ax.set_ylabel('loss value')
ax.set_xlabel('iteration times')
ax.set_title('DSVI DGP loss-train %d point(s)/ test %d point(s)' % (len(train_y),len(test_y)))
plt.show()

# show the worst parameter
print('\033[1mthe worst parameter\033[0m')
print(
    '    hidden_lengthscale: ',worst_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', worst_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',worst_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',worst_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',worst_hyperparams['likelihood_noise'],
    ':\n   num_hidden_dims', worst_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',worst_hyperparams['num_inducing']
)
print('    loss:', worst_result['loss'],
    ': test rmse: ', worst_result['test rmse'],
    '; \033[1mtrain rmse: ', worst_result['train rmse'],'\033[0m')
print('*************************************')

# show the best and the second-best parameter and third-best parameter and the fourth
print('method',method)
print('\033[1mthe best parameter\033[0m')
print(
    '    hidden_lengthscale: ',best_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', best_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',best_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',best_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',best_hyperparams['likelihood_noise'],
    ':\n   num_hidden_dims', best_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',best_hyperparams['num_inducing']
)
print('    loss:', best_result['loss'],
    ': test rmse: ', best_result['test rmse'],
    '; \033[1mtrain rmse: ', best_result['train rmse'],'\033[0m')

print('\033[1mthe second best parameter\033[0m')
print(
    '    hidden_lengthscale: ',second_best_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', second_best_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',second_best_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',second_best_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',second_best_hyperparams['likelihood_noise'],
    ':\n   num_hidden_dims', second_best_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',second_best_hyperparams['num_inducing'])
print('    loss:', second_best_result['loss'],
    ': test rmse: ', second_best_result['test rmse'],
    '; \033[1mtrain rmse: ', second_best_result['train rmse'],'\033[0m')

print('\033[1mthe third best parameter\033[0m')
print(
    '    hidden_lengthscale: ',third_best_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', third_best_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',third_best_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',third_best_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',third_best_hyperparams['likelihood_noise'],
    ':\n   num_hidden_dims', third_best_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',third_best_hyperparams['num_inducing'])
print('    loss:', third_best_result['loss'],
    ': test rmse: ', third_best_result['test rmse'],
    '; \033[1mtrain rmse: ', third_best_result['train rmse'],'\033[0m')

print('\033[1mthe fourth best parameter\033[0m')
print(
    '    hidden_lengthscale: ',fourth_best_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', fourth_best_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',fourth_best_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',fourth_best_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',fourth_best_hyperparams['likelihood_noise'],
    ';\n   num_hidden_dims', fourth_best_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',fourth_best_hyperparams['num_inducing'])
print('    loss:', fourth_best_result['loss'],
    ': test rmse: ', fourth_best_result['test rmse'],
    '; \033[1mtrain rmse: ', fourth_best_result['train rmse'],'\033[0m')

print('\033[1mthe fifth best parameter\033[0m')
print(
    '    hidden_lengthscale: ',fifth_best_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', fifth_best_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',fifth_best_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',fifth_best_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',fifth_best_hyperparams['likelihood_noise'],
    ':\n   num_hidden_dims', fifth_best_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',fifth_best_hyperparams['num_inducing'])
print('    loss:', fifth_best_result['loss'],
    ': test rmse: ', fifth_best_result['test rmse'],
    '; \033[1mtrain rmse: ', fifth_best_result['train rmse'],'\033[0m')

print('\033[1mthe sixth best parameter\033[0m')
print(
    '    hidden_lengthscale: ',sixth_best_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', sixth_best_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',sixth_best_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',sixth_best_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',sixth_best_hyperparams['likelihood_noise'],
    ':\n   num_hidden_dims', sixth_best_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',sixth_best_hyperparams['num_inducing'])
print('    loss:', sixth_best_result['loss'],
    ': test rmse: ', sixth_best_result['test rmse'],
    '; \033[1mtrain rmse: ', sixth_best_result['train rmse'],'\033[0m')

print('\033[1mthe seventh best parameter\033[0m')
print(
    '    hidden_lengthscale: ',seventh_best_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', seventh_best_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',seventh_best_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',seventh_best_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',seventh_best_hyperparams['likelihood_noise'],
    ':\n   num_hidden_dims', seventh_best_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',seventh_best_hyperparams['num_inducing'])
print('    loss:', seventh_best_result['loss'],
    ': test rmse: ', seventh_best_result['test rmse'],
    '; \033[1mtrain rmse: ', seventh_best_result['train rmse'],'\033[0m')

print('\033[1mthe eighth best parameter\033[0m')
print(
    '    hidden_lengthscale: ',eighth_best_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', eighth_best_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',eighth_best_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',eighth_best_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',eighth_best_hyperparams['likelihood_noise'],
    ':\n   num_hidden_dims', eighth_best_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',eighth_best_hyperparams['num_inducing'])
print('    loss:', eighth_best_result['loss'],
    ': test rmse: ', eighth_best_result['test rmse'],
    '; \033[1mtrain rmse: ', eighth_best_result['train rmse'],'\033[0m')

print('\033[1mthe ninth best parameter\033[0m')
print(
    '    hidden_lengthscale: ',ninth_best_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', ninth_best_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',ninth_best_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',ninth_best_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',ninth_best_hyperparams['likelihood_noise'],
    ':\n   num_hidden_dims', ninth_best_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',ninth_best_hyperparams['num_inducing'])
print('    loss:', ninth_best_result['loss'],
    ': test rmse: ', ninth_best_result['test rmse'],
    '; \033[1mtrain rmse: ', ninth_best_result['train rmse'],'\033[0m')

print('\033[1mthe tenth best parameter\033[0m')
print(
    '    hidden_lengthscale: ',tenth_best_hyperparams['hidden_layer_lengthscale'],
    ';\n   hidden_outputscale: ', tenth_best_hyperparams['hidden_layer_outputscale'],
    ';\n   last_lengthscale: ',tenth_best_hyperparams['last_layer_lengthscale'],
    ';\n   last_outputscale: ',tenth_best_hyperparams['last_layer_outputscale'],
    ';\n   noise: ',tenth_best_hyperparams['likelihood_noise'],
    ':\n   num_hidden_dims', tenth_best_hyperparams['num_hidden_dim'],
    ':\n   num_inducing: ',tenth_best_hyperparams['num_inducing'])
print('    loss:', tenth_best_result['loss'],
    ': test rmse: ', tenth_best_result['test rmse'],
    '; \033[1mtrain rmse: ', tenth_best_result['train rmse'],'\033[0m')