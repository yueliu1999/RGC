import os
import argparse
import warnings
from utils import *
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch_scatter import scatter
from model import my_model, my_Q_net
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")

# E net
parser.add_argument('--E_epochs', type=int, default=400, help='Number of epochs to train E.')
parser.add_argument('--n_input', type=int, default=1000, help='Number of units in hidden layer 1.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')

# Q net
parser.add_argument('--Q_epochs', type=int, default=30, help='Number of epochs to train Q.')
parser.add_argument('--epsilon', type=float, default=0.5, help='Greedy rate.')
parser.add_argument('--replay_buffer_size', type=float, default=50, help='Replay buffer size')
parser.add_argument('--Q_lr', type=float, default=1e-3, help='Initial learning rate.')


args = parser.parse_args()

device = "cuda:0"
file_name = "result.csv"
for args.dataset in ["bat", "eat", "cora", "citeseer"]:
    # "amap",
    file = open(file_name, "a+")
    print(args.dataset, file=file)
    file.close()

    if args.dataset == 'cora':
        args.cluster_num = 7
        args.gnnlayers = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = [1500]
        args.epsilon = 0.5
        args.replay_buffer_size = 40

    elif args.dataset == 'citeseer':
        args.cluster_num = 6
        args.gnnlayers = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = [1500]
        args.epsilon = 0.7
        args.replay_buffer_size = 50

    elif args.dataset == 'amap':
        args.cluster_num = 8
        args.gnnlayers = 3
        args.lr = 1e-5
        args.n_input = -1
        args.dims = [500]
        args.epsilon = 0.7
        args.replay_buffer_size = 50

    elif args.dataset == 'bat':
        args.cluster_num = 4
        args.gnnlayers = 6
        args.lr = 1e-3
        args.n_input = -1
        args.dims = [1500]
        args.epsilon = 0.3
        args.replay_buffer_size = 30

    elif args.dataset == 'eat':
        args.cluster_num = 4
        args.gnnlayers = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = [1500]
        args.epsilon = 0.7
        args.replay_buffer_size = 40

    nmi_list = []
    ari_list = []
    k_list = []
    # init
    for seed in range(10):
        setup_seed(seed)
        X, y, A = load_graph_data(args.dataset, show_details=False)
        features = X
        true_labels = y
        adj = sp.csr_matrix(A)

        if args.n_input != -1:
            pca = PCA(n_components=args.n_input)
            features = pca.fit_transform(features)

        A = torch.tensor(adj.todense()).float().to(device)

        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        print('Laplacian Smoothing...')
        adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
        sm_fea_s = sp.csr_matrix(features).toarray()

        path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
        if os.path.exists(path):
            sm_fea_s = np.load(path, allow_pickle=True)
        else:
            for a in adj_norm_s:
                sm_fea_s = a.dot(sm_fea_s)
            np.save(path, sm_fea_s, allow_pickle=True)

        # X
        sm_fea_s = torch.FloatTensor(sm_fea_s)
        adj_1st = (adj + sp.eye(adj.shape[0])).toarray()

        # test
        # best_nmi = 0
        # best_ari = 0
        args.cluster_num = np.random.randint(0, 9) + 2

        # init clustering
        _, _, predict_labels, _, _ = clustering(sm_fea_s.detach(), true_labels, args.cluster_num, device=device)
        best_nmi = 0
        best_ari = 0
        # MLP
        if args.dataset == "citeseer":
            model = my_model([sm_fea_s.shape[1]] + args.dims, act="sigmoid")
        else:
            model = my_model([sm_fea_s.shape[1]] + args.dims)
        Q_net = my_Q_net(args.dims + [256, 9]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer_Q = optim.Adam(Q_net.parameters(), lr=args.Q_lr)

        model.to(device)
        inx = sm_fea_s.to(device)
        inx_origin = torch.FloatTensor(features).to(device)

        A_label = torch.FloatTensor(adj_1st).to(device)

        target = A_label
        mask = torch.ones([target.shape[0] * 2, target.shape[0] * 2]).to(device)
        mask -= torch.diag_embed(torch.diag(mask))

        tmp_epsilon = args.epsilon
        epsilon_step = (1 - tmp_epsilon) / args.E_epochs
        replay_buffer = []

        print('Start Training...')
        for epoch in tqdm(range(args.E_epochs)):
            model.train()
            Q_net.eval()
            optimizer.zero_grad()
            z1, z2 = model(inx)

            z1_z2 = torch.cat([z1, z2], dim=0)
            S = z1_z2 @ z1_z2.T

            # pos neg weight
            pos_neg = mask * torch.exp(S)

            pos = torch.cat([torch.diag(S, target.shape[0]), torch.diag(S, -target.shape[0])], dim=0)
            # pos weight
            pos = torch.exp(pos)

            neg = (torch.sum(pos_neg, dim=1) - pos)

            infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * target.shape[0])
            loss = infoNEC
            state = (z1 + z2) / 2

            cluster_state = scatter(state, torch.tensor(predict_labels).to(device), dim=0, reduce="mean")

            rand = False

            # do action by random choose
            if random.random() > args.epsilon:
                action = np.random.randint(0, 9)
                rand = True
            # do action by Q-net
            else:
                action = int(Q_net(state, cluster_state).mean(0).argmax())

            args.cluster_num = action + 2
            nmi, ari, predict_labels, centers, dis = clustering(state.detach(), true_labels, args.cluster_num, device=device)
            dis = (state.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1) + 1

            q = dis / (dis.sum(-1).reshape(-1, 1))
            p = q.pow(2) / q.sum(0).reshape(1, -1)
            p = p / p.sum(-1).reshape(-1, 1)
            pq_loss = F.kl_div(q.log(), p)
            loss += 10 * pq_loss

            if nmi >= best_nmi and rand == False:
                best_nmi = nmi
                best_ari = ari
                best_cluster = args.cluster_num

            loss.backward()
            optimizer.step()

            # next state
            model.eval()
            z1, z2 = model(inx)
            next_state = (z1 + z2) / 2

            next_cluster_state = scatter(next_state, torch.tensor(predict_labels).to(device), dim=0, reduce="mean")

            center_dis = (centers.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1).mean()
            reward = center_dis.detach() - torch.min(dis, dim=1).values.mean().detach()

            replay_buffer.append([[state.detach(), cluster_state.detach()], action,
                                  [next_state.detach(), next_cluster_state.detach()], reward])

            tmp_epsilon += epsilon_step

            # replay_buffer full: train Q network
            if len(replay_buffer) >= args.replay_buffer_size:
                for it in range(args.Q_epochs):
                    model.eval()
                    Q_net.train()
                    optimizer_Q.zero_grad()
                    idx = list(range(args.replay_buffer_size))
                    np.random.shuffle(idx)
                    loss_Q = 0
                    for i in idx:
                        s = replay_buffer[i][0][0]
                        s_c = replay_buffer[i][0][1]
                        a = replay_buffer[i][1]
                        s_new = replay_buffer[i][2][0]
                        s_new_c = replay_buffer[i][2][1]
                        r = replay_buffer[i][3]
                        Q_value = Q_net(s, s_c).mean(0)[a]
                        y = r + 0.1 * Q_net(s_new, s_new_c).mean(0).max()
                        loss_Q += (y - Q_value) ** 2
                    loss_Q /= len(idx)
                    # loss_Q = loss_Q ** 0.5
                    # MSE loss
                    loss_Q.backward()
                    optimizer_Q.step()
                # cleaning up
                replay_buffer = []

        file = open(file_name, "a+")
        print(best_cluster, best_nmi, best_ari, file=file)
        file.close()
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)
        k_list.append(best_cluster)

        tqdm.write("Optimization Finished!")
        tqdm.write('best_nmi: {}, best_ari: {}'.format(best_nmi, best_ari))

    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    k_list = np.array(k_list)

    file = open(file_name, "a+")
    print(args.gnnlayers, args.lr, file=file)
    print(k_list.mean(), k_list.std(), file=file)
    print(nmi_list.mean(), nmi_list.std(), file=file)
    print(ari_list.mean(), ari_list.std(), file=file)
    file.close()
