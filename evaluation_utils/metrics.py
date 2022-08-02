'''
    This script defines the metrics used to evaluate a submission.
    The metrics are consistent with the ones used in the evaluation 
    table of streaming and iid protocols, i.e., Table 1, in the paper. 
    More information the two protocols and metrics:
    https://clear-benchmark.github.io/
'''
import os
import matplotlib.pyplot as plt


# Average of lower triangle + diagonal (evaluate accuracy on seen tasks)
def accuracy(M):
    r, _ = M.shape
    res = [M[i, j] for i in range(r) for j in range(i+1)]
    return sum(res) / len(res)


# Diagonal average (evaluates accuracy on the current task)
def in_domain(M):
    r, _ = M.shape
    return sum([M[i, i] for i in range(r)]) / r


# Superdiagonal average (evaluate on the immediate next time period)
def next_domain(M):
    r, _ = M.shape
    return sum([M[i, i+1] for i in range(r-1)]) / (r-1)


# Upper trianglar average (evaluate generalation)
def forward_transfer(M):
    r, _ = M.shape
    res = [M[i, j] for i in range(r) for j in range(i+1, r)]
    return sum(res) / len(res)


# Lower triangular average (evaluate learning without forgetting)
def backward_transfer(M):
    r, _ = M.shape
    res = [M[i, j] for i in range(r) for j in range(i)]
    return sum(res) / len(res)


def format_score(s):
    return f"{s:.2%}"


# Generate heatmap for the accuracy matrix
def plot_2d_matrix(matrix, label_ticks, title, save_name, save_path='.', min_acc=None, max_acc=None):
    matrix_score = matrix

    plt.figure(figsize=(10,10))
    x = ["Bucket " + n for n in label_ticks]
    y = ['Model ' + n for n in label_ticks]
    p = plt.imshow(matrix, interpolation='none', cmap=f'Blues', vmin=min_acc, vmax=max_acc)

    plt.xticks(range(len(x)), x, fontsize=11, rotation = -90)
    plt.yticks(range(len(y)), y, fontsize=11)
    plt.title(title, fontsize=15)
    for i in range(len(x)):
        for j in range(len(y)):
            text = plt.text(j, i, format_score(matrix_score[i, j]),
                       ha="center", va="center", color="black")
    os.makedirs(save_path,exist_ok=True)
    plt.savefig(os.path.join(save_path,'{}.png'.format(save_name)))
