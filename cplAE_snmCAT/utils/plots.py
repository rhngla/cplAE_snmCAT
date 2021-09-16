import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def scatter3(X,col,xlims=(-3,3),ylims=(-3,3),zlims=(-3,3),fig=None):
    
    if fig is None:
        fig = plt.figure(figsize=(4,4))
        

    plt.ion()    
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X[:,0], X[:,1], X[:,2],s=1,alpha=1,c=col)

    ax.set_xticks([])
    ax.set_zticks([])
    ax.set_yticks([])
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_zlim(zlims[0],zlims[1])

    plt.axis('on')
    ax.set_frame_on(False)
    plt.tight_layout()
    return ax,sc


def discrete_to_color(label_array, label_order=None, palette="husl"):
    """Get colors that indicate distinct discrete labels

    Args:
        label_array (np.array or list): array of discrete labels.
        label_order (np.array or list): labels will be assigned colors in this order. 
            remaining labels will be given a single color
        palette: seaborn palette name. husl gives an arbitrary number of colors.
 
    Returns:
        color_array: array of colors according to label_array (n_samples x rgb)
    """
    unique_labels = np.unique(label_array)
    cmap = {}
    if label_order is not None:
        unspecified_labels = np.setdiff1d(unique_labels, label_order)
        unique_labels = np.array(label_order)
        cmap = {key: (0.7, 0.7, 0.7) for key in unspecified_labels}

    unique_colors = sns.color_palette(palette, np.size(unique_labels))
    cmap.update(dict(zip(unique_labels, unique_colors)))

    def colorize(x): return cmap[x]
    colorize_vec = np.vectorize(colorize)
    color_array = np.transpose(colorize_vec(label_array))
    return color_array
