'''Visualize Collatz numbers.
'''

import numpy as np
import time


def CollatzNext(n):
    '''Compute the next Collatz number

    Args:
        n (int): current number
    Returns:
        n//2 if n is even, 3*n+1 otherwise
    '''
    return n//2 if n%2==0 else 3*n+1


def CollatzSeq(n, cache=None):
    '''Get the Collatz sequence from a give positive int down to 1

    Args:
        n (int): current number.
    Kwargs:
        cache (dict): cache to store known sequences.
                      Key: give poistive int.
                      Value: Collatz sequence starting from the key. E.g.
                      cache[4] = [4, 2, 1].
    Returns:
        res (list): Collatz sequence starting from <n>.
        cache (dict): updated cache.
    '''

    if n<1 or int(n) != n:
        raise Exception("<n> needs to be a positive integer >= 1.")

    if cache is None:
        cache={}

    if n in cache:
        return cache[n]

    res=[n]
    cur=n
    while res[-1]!=1:
        new=CollatzNext(cur)
        # see new number in cache or not, if so, extend the cached seq
        if new in cache:
            res.extend(cache[new])
        else:
            res.append(new)
            cur=new

    # update cache
    cache[n]=res

    return res, cache


def CollatzSeqCoords(seq, cache=None):
    '''Compute coordinates of Collatz sequence for plotting

    Args:
        seq (list): a given Collatz sequence, with last element being 1.
    Kwargs:
        cache (dict): cache to store known sequence coordinates.
                      Key: give poistive int.
                      Value: Collatz sequence coordinates starting from the key.
    Returns:
        new_coords (ndarray): Nx3 ndarray of the coordinates of the Collatz sequence.
                              N is the length of <seq>.
                              Column 1: x-coordinates.
                              Column 2: y-coordinates.
                              Column 3: angle of line in degrees.
        cache (dict): updated cache.
    '''

    dl=1.0    # line segment length
    theta0=0  # line orientation of the starting point/root point (i.e. 1)
    dtheta_odd=-np.pi*1.618 # rotation angle for an odd Collatz number, in degrees
    dtheta_even=np.pi       # rotation angle for an even Collatz number, in degrees

    if cache is None:
        cache={}

    if seq[0] in cache:
        return cache[seq[0]]

    # the root point 1
    if len(seq)==1 and seq[0]==1:
        res=np.array([[0, 0, theta0]])
        cache[1]=res
        return res, cache

    # find the known sub-sequence
    # E.g. seq = [8, 4, 2, 1]
    # 2 in cache, then known sub-sequence is [2, 1]
    cur=0
    while True:
        if seq[cur] in cache:
            break
        cur+=1

    found_coords=cache[seq[cur]]   # coordinates of the known sequence
    remain_seq=seq[:cur][::-1]     # new numbers not known in cache, e.g. [4, 8]
    last_number=seq[cur]           # last known Collatz number, e.g. 2
    last=found_coords[0]           # coordinate of the last known Collatz number
    new_coords=[]                  # new coordinates

    def sigmoid(x):
        return 1./(1+np.exp(-x))

    for ii, nii in enumerate(remain_seq):

        lx, ly, ltheta=last  # last known point
        # compute rotation angle
        if nii%2==0:
            dtheta=dtheta_even
        else:
            dtheta=dtheta_odd
        dtheta=dtheta*(1+sigmoid(abs(nii-last_number))) # scale up rotation

        # compute the coordinate for the current point
        newtheta=ltheta+dtheta
        newxii=lx+dl*np.cos(newtheta*np.pi/180)
        newyii=ly+dl*np.sin(newtheta*np.pi/180)
        new_coords.append((newxii, newyii, newtheta))

        # update the last known point
        last=(newxii, newyii, newtheta)
        last_number=nii

    new_coords=np.r_[np.array(new_coords)[::-1], found_coords]

    # update cache
    cache[seq[0]]=new_coords

    return new_coords, cache


def update_all(i, line_params, ax):

    results=[]
    for ii in range(len(line_params)):
        xx, yy, line=line_params[ii]
        line.set_data(xx[:i], yy[:i])
        results.append(line)
    return results


#-------------Main---------------------------------
if __name__=='__main__':

    numbers=range(1,12000)

    #------------Compute Collatz sequences------------
    cache={}
    t1=time.time()
    for i in numbers:
        aa, cache=CollatzSeq(i, cache=cache)
    t2=time.time()
    print('time=',t2-t1)

    #-------------Compute line coordinates-------------
    coord_cache={}
    t1=time.time()
    for i in numbers:
        sii=cache[i]
        _, coord_cache=CollatzSeqCoords(sii, cache=coord_cache)
    t2=time.time()
    print('time=',t2-t1)

    #-------------------Plot------------------------
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.animation import FuncAnimation
    figure=plt.figure(figsize=(12,10),dpi=50)
    ax=figure.add_subplot(111)

    cmap=plt.cm.autumn
    cmap=plt.cm.terrain
    cmap=plt.cm.gnuplot
    #cmap=plt.cm.rainbow

    print('\n# <testcollatz>: Creating plot...')

    '''
    #-----------------Create animation-----------------
    lines=[]
    frame_num=0
    xmax=0; xmin=0
    ymax=0; ymin=0

    for ii, nii in enumerate(numbers):
        print('ii=',ii)

        xii=coord_cache[nii][::-1, 0]
        yii=coord_cache[nii][::-1, 1]
        frame_num=max(frame_num, len(yii))
        xmax=max(xmax, np.max(xii))
        ymax=max(ymax, np.max(yii))
        xmin=min(xmin, np.min(xii))
        ymin=min(ymin, np.min(yii))

        # select a random color
        colorii=(float(ii)/numbers[-1]+np.random.rand())%1.
        lii,=ax.plot([], [], color=cmap(colorii), linewidth=16)
        lii.set_path_effects([pe.Normal(), pe.Stroke(linewidth=3, foreground='w')])
        lii.set_solid_capstyle('round')
        lii.set_solid_joinstyle('round')

        lines.append([xii, yii, lii])

    ax.set_xlim(xmin-(xmax-xmin)*0.03, xmax*1.02)
    ax.set_ylim(ymin-(ymax-ymin)*0.03, ymax*1.02)
    plt.axis('off')

    print('\n# <testcollatz>: Rendering animation ...')
    anim=FuncAnimation(figure, update_all, frames=frame_num,
            fargs=(lines, ax),
            interval=8,
            repeat=False,
            blit=True)

    #figure.show()
    anim.save('Collatz2.mp4', fps=30)
    '''


    #----------------Plot final outputs----------------
    for ii in numbers:

        xii=coord_cache[ii][:, 0]
        yii=coord_cache[ii][:, 1]

        # select a random color
        colorii=(float(ii)/numbers[-1]+np.random.rand())%1.
        lii,=ax.plot(xii, yii, linestyle='-', color=cmap(colorii),
                linewidth=16)
        lii.set_path_effects([pe.Normal(), pe.Stroke(linewidth=3, foreground='w')])
        lii.set_solid_capstyle('round')
        lii.set_solid_joinstyle('round')

    ax.set_frame_on(False)
    ax.axis('off')

    figure.show()
    figure.savefig('Collatz.png', dpi=100)







